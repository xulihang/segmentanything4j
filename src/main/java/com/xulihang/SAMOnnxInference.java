package com.xulihang;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import ai.onnxruntime.*;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;

public class SAMOnnxInference {
    static {
        nu.pattern.OpenCV.loadLocally();
    }

    private OrtEnvironment env;
    private OrtSession session;
    private OrtSession encoderSession;
    private boolean needEmbeddings;
    private String rawInputName;
    private int targetSize = 1024;
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std = {0.229f, 0.224f, 0.225f};

    public SAMOnnxInference(String modelPath, String encoderPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();

        // 配置Session选项
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);

        this.session = env.createSession(modelPath, sessionOptions);

        // 检查输入需求
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        needEmbeddings = inputInfo.containsKey("image_embeddings");

        if (needEmbeddings) {
            if (encoderPath == null || !new File(encoderPath).exists()) {
                throw new RuntimeException("ONNX model expects 'image_embeddings' but no encoder ONNX found.");
            }
            this.encoderSession = env.createSession(encoderPath, sessionOptions);
            System.out.println("Encoder session loaded");
        } else {
            // 查找图像输入名称
            for (String name : inputInfo.keySet()) {
                if (name.equals("image") || name.equals("images") ||
                        name.equals("pixel_values") || name.equals("input_image")) {
                    rawInputName = name;
                    break;
                }
            }
            if (rawInputName == null) {
                rawInputName = inputInfo.keySet().iterator().next();
            }
            System.out.println("Using raw input name: " + rawInputName);
        }
    }

    public static class PreprocessResult {
        public OnnxTensor tensor;
        public float scale;
        public Size originalSize;
        public Size resizedSize;

        public PreprocessResult(OnnxTensor tensor, float scale, Size originalSize, Size resizedSize) {
            this.tensor = tensor;
            this.scale = scale;
            this.originalSize = originalSize;
            this.resizedSize = resizedSize;
        }
    }

    public PreprocessResult preprocessForEncoder(Mat image) throws OrtException {
        int h0 = image.rows();
        int w0 = image.cols();

        float scale = (float) targetSize / Math.max(h0, w0);
        int newW = (int) (w0 * scale);
        int newH = (int) (h0 * scale);

        System.out.printf("Original size: %dx%d, Resized: %dx%d, Scale: %.4f%n",
                w0, h0, newW, newH, scale);

        // 调整大小
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_LINEAR);

        // 填充到正方形 (右下填充)
        Mat canvas = Mat.zeros(targetSize, targetSize, CvType.CV_8UC3);
        Mat roi = canvas.submat(0, newH, 0, newW);
        resized.copyTo(roi);

        // 转换为float 0-1
        Mat imgFloat = new Mat();
        canvas.convertTo(imgFloat, CvType.CV_32F, 1.0 / 255.0);

        // 应用归一化 - 逐通道处理
        List<Mat> channels = new ArrayList<>(3);
        Core.split(imgFloat, channels);

        for (int i = 0; i < 3; i++) {
            Core.subtract(channels.get(i), new Scalar(mean[i]), channels.get(i));
            Core.divide(channels.get(i), new Scalar(std[i]), channels.get(i));
        }

        Core.merge(channels, imgFloat);

        // HWC -> CHW
        Mat chw = new Mat();
        Core.transpose(imgFloat, chw);

        // 创建tensor数据 [1, 3, H, W]
        float[] tensorData = new float[3 * targetSize * targetSize];
        chw.get(0, 0, tensorData);

        long[] shape = {1, 3, (long)targetSize, (long)targetSize};
        OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape);

        return new PreprocessResult(tensor, scale, new Size(w0, h0), new Size(newW, newH));
    }

    public float[][] transformCoords(float[][] coords, Size origSize, float scale) {
        float[][] transformed = new float[coords.length][2];
        for (int i = 0; i < coords.length; i++) {
            // 注意：Python版本是 coords * scale
            transformed[i][0] = coords[i][0] * scale;
            transformed[i][1] = coords[i][1] * scale;
        }
        return transformed;
    }

    public OnnxTensor getImageEmbeddings(PreprocessResult preprocessed) throws OrtException {
        if (encoderSession == null) {
            throw new IllegalStateException("Encoder session not initialized");
        }

        String inputName = encoderSession.getInputNames().iterator().next();
        OrtSession.Result result = encoderSession.run(Collections.singletonMap(inputName, preprocessed.tensor));

        OnnxValue embeddingValue = result.get(0);
        return (OnnxTensor) embeddingValue;
    }

    public Mat infer(float[] box, float[][] points, float[] pointLabels, Mat image) throws OrtException {
        // 预处理图像
        PreprocessResult preprocessed = preprocessForEncoder(image);

        // 准备提示数据
        float[][] inputBoxCoords = {{box[0], box[1]}, {box[2], box[3]}};
        float[] boxLabels = {2, 3};

        // 合并点和框坐标
        float[][] allCoords;
        float[] allLabels;

        if (points == null || points.length == 0) {
            allCoords = inputBoxCoords;
            allLabels = boxLabels;
        } else {
            allCoords = new float[points.length + 2][2];
            allLabels = new float[points.length + 2];

            // 复制点
            System.arraycopy(points, 0, allCoords, 0, points.length);
            System.arraycopy(pointLabels, 0, allLabels, 0, pointLabels.length);

            // 添加框坐标
            allCoords[points.length] = inputBoxCoords[0];
            allCoords[points.length + 1] = inputBoxCoords[1];
            allLabels[points.length] = 2;
            allLabels[points.length + 1] = 3;
        }

        // 转换坐标到预处理后的空间
        float[][] transformedCoords = transformCoords(allCoords, preprocessed.originalSize, preprocessed.scale);

        System.out.printf("Transformed coords: %s%n", Arrays.deepToString(transformedCoords));

        // 构建输入字典
        Map<String, OnnxTensor> inputs = new HashMap<>();

        if (needEmbeddings) {
            OnnxTensor embeddings = getImageEmbeddings(preprocessed);
            inputs.put("image_embeddings", embeddings);
        } else {
            inputs.put(rawInputName, preprocessed.tensor);
        }

        // 准备点坐标输入 [1, N, 2]
        float[][][] pointCoordsArray = {transformedCoords};
        long[] pointCoordsShape = {1, (long)transformedCoords.length, 2};
        FloatBuffer pointCoordsBuffer = FloatBuffer.allocate(transformedCoords.length * 2);
        for (float[] coord : transformedCoords) {
            pointCoordsBuffer.put(coord);
        }
        pointCoordsBuffer.rewind();
        inputs.put("point_coords", OnnxTensor.createTensor(env, pointCoordsBuffer, pointCoordsShape));

        // 准备点标签输入 [1, N]
        float[][] pointLabelsArray = {allLabels};
        long[] pointLabelsShape = {1, (long)allLabels.length};
        FloatBuffer pointLabelsBuffer = FloatBuffer.allocate(allLabels.length);
        for (float label : allLabels) {
            pointLabelsBuffer.put(label);
        }
        pointLabelsBuffer.rewind();
        inputs.put("point_labels", OnnxTensor.createTensor(env, pointLabelsBuffer, pointLabelsShape));

        // 掩码输入 [1, 1, 256, 256] - 全零
        float[] maskInputData = new float[1 * 1 * 256 * 256];
        long[] maskInputShape = {1, 1, 256, 256};
        inputs.put("mask_input", OnnxTensor.createTensor(env,
                FloatBuffer.wrap(maskInputData), maskInputShape));

        // has_mask_input [1] - 零
        float[] hasMaskInput = {0};
        long[] hasMaskShape = {1};
        inputs.put("has_mask_input", OnnxTensor.createTensor(env,
                FloatBuffer.wrap(hasMaskInput), hasMaskShape));

        // 原始图像尺寸 [2]
        float[] origImSize = {(float)preprocessed.originalSize.height, (float)preprocessed.originalSize.width};
        long[] origImSizeShape = {2};
        inputs.put("orig_im_size", OnnxTensor.createTensor(env,
                FloatBuffer.wrap(origImSize), origImSizeShape));

        System.out.println("Running inference with inputs: " + inputs.keySet());

        // 运行推理
        long startTime = System.currentTimeMillis();
        OrtSession.Result results = session.run(inputs);
        long endTime = System.currentTimeMillis();
        System.out.printf("Inference time: %d ms%n", endTime - startTime);

        // 处理输出
        OnnxValue masksValue = results.get(0);
        Mat mask = processMaskOutput(masksValue, preprocessed.originalSize);

        // 清理tensor
        for (OnnxTensor tensor : inputs.values()) {
            tensor.close();
        }
        preprocessed.tensor.close();

        return mask;
    }

    private Mat processMaskOutput(OnnxValue masksValue, Size originalSize) throws OrtException {
        Object value = masksValue.getValue();
        Mat mask;

        if (value instanceof float[][][][]) {
            float[][][][] maskData = (float[][][][]) value;
            int batch = maskData.length;
            int channel = maskData[0].length;
            int height = maskData[0][0].length;
            int width = maskData[0][0][0].length;

            System.out.printf("Mask shape: [%d, %d, %d, %d]%n", batch, channel, height, width);

            // 取第一个batch和channel
            mask = new Mat(height, width, CvType.CV_32F);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    mask.put(i, j, maskData[0][0][i][j]);
                }
            }
        } else {
            throw new RuntimeException("Unexpected mask output format: " + value.getClass());
        }

        // 应用阈值
        float maskThreshold = 0.0f;
        Mat binaryMask = new Mat();
        Core.compare(mask, new Scalar(maskThreshold), binaryMask, Core.CMP_GT);

        // 转换为8UC1
        Mat mask8u = new Mat();
        binaryMask.convertTo(mask8u, CvType.CV_8UC1, 255);

        // 调整到原始尺寸 - 使用最近邻插值
        Mat resizedMask = new Mat();
        Imgproc.resize(mask8u, resizedMask,
                new Size(originalSize.width, originalSize.height),
                0, 0, Imgproc.INTER_NEAREST);

        return resizedMask;
    }

    public void close() throws OrtException {
        if (session != null) session.close();
        if (encoderSession != null) encoderSession.close();
        if (env != null) env.close();
    }
}