package com.xulihang;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import ai.onnxruntime.*;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.List;

public class SAMOnnxInference {
    static {
        // 加载OpenCV本地库
        nu.pattern.OpenCV.loadLocally();
    }

    private OrtEnvironment env;
    private OrtSession session;
    private OrtSession encoderSession;
    private boolean needEmbeddings;
    private String rawInputName;
    private int targetSize = 1024;
    private float[] mean = {0.485f, 0.456f, 0.406f};
    private float[] std = {0.229f, 0.224f, 0.225f};

    public SAMOnnxInference(String modelPath, String encoderPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());

        // 检查是否需要image_embeddings
        List<String> inputNames = new ArrayList<>();
        for (NodeInfo input : session.getInputInfo().values()) {
            inputNames.add(input.getName());
        }
        needEmbeddings = inputNames.contains("image_embeddings");

        if (needEmbeddings) {
            if (encoderPath == null || !new File(encoderPath).exists()) {
                throw new RuntimeException("ONNX model expects 'image_embeddings' but no encoder ONNX found.");
            }
            this.encoderSession = env.createSession(encoderPath, new OrtSession.SessionOptions());
        } else {
            // 查找原始图像输入名称
            for (String name : inputNames) {
                if (name.equals("image") || name.equals("images") ||
                        name.equals("pixel_values") || name.equals("input_image")) {
                    rawInputName = name;
                    break;
                }
            }
            if (rawInputName == null) {
                rawInputName = session.getInputNames().iterator().next();
            }
        }
    }

    public static class PreprocessResult {
        public OnnxTensor tensor;
        public float scale;
        public Size paddedSize;

        public PreprocessResult(OnnxTensor tensor, float scale, Size paddedSize) {
            this.tensor = tensor;
            this.scale = scale;
            this.paddedSize = paddedSize;
        }
    }

    public PreprocessResult preprocessForEncoder(Mat image) throws OrtException {
        int targetSize = this.targetSize;
        int h0 = image.rows();
        int w0 = image.cols();

        float scale = (float) targetSize / Math.max(h0, w0);
        int newW = (int) (w0 * scale);
        int newH = (int) (h0 * scale);

        // 调整大小
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(newW, newH));

        // 填充到正方形
        Mat canvas = Mat.zeros(targetSize, targetSize, CvType.CV_8UC3);
        Mat roi = canvas.submat(0, newH, 0, newW);
        resized.copyTo(roi);

        // 转换为float并归一化
        Mat imgFloat = new Mat();
        canvas.convertTo(imgFloat, CvType.CV_32F, 1.0 / 255.0);

        // 应用归一化
        List<Mat> channels = new ArrayList<>();
        Core.split(imgFloat, channels);

        for (int i = 0; i < 3; i++) {
            Core.subtract(channels.get(i), new Scalar(mean[i]), channels.get(i));
            Core.divide(channels.get(i), new Scalar(std[i]), channels.get(i));
        }

        Core.merge(channels, imgFloat);

        // HWC -> CHW
        Mat chw = new Mat();
        Core.transpose(imgFloat, chw);

        // 创建tensor
        float[][][] tensorData = new float[1][3][targetSize * targetSize];
        float[] flatData = new float[3 * targetSize * targetSize];
        chw.get(0, 0, flatData);

        // 重新组织数据为 [1, 3, H, W]
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < targetSize; h++) {
                for (int w = 0; w < targetSize; w++) {
                    int srcIdx = c * targetSize * targetSize + h * targetSize + w;
                    tensorData[0][c][h * targetSize + w] = flatData[srcIdx];
                }
            }
        }

        long[] shape = {1, 3, (long)targetSize, (long)targetSize};
        OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatData), shape);

        return new PreprocessResult(tensor, scale, new Size(newW, newH));
    }

    public float[][] transformCoords(float[][] coords, Size origSize, float scale) {
        float[][] transformed = new float[coords.length][2];
        for (int i = 0; i < coords.length; i++) {
            transformed[i][0] = coords[i][0] * scale;
            transformed[i][1] = coords[i][1] * scale;
        }
        return transformed;
    }

    public void showMask(Mat mask, Mat image) {
        // 创建带透明度的掩码可视化
        Mat maskColor = new Mat(mask.size(), CvType.CV_8UC3, new Scalar(30, 144, 255));
        Mat maskAlpha = new Mat(mask.size(), CvType.CV_32F);
        mask.convertTo(maskAlpha, CvType.CV_32F, 0.6);

        // 将掩码应用到原图
        Mat result = new Mat();
        image.copyTo(result);

        for (int i = 0; i < mask.rows(); i++) {
            for (int j = 0; j < mask.cols(); j++) {
                if (mask.get(i, j)[0] > 0) {
                    double[] color = maskColor.get(i, j);
                    double alpha = maskAlpha.get(i, j)[0];
                    double[] original = result.get(i, j);

                    double[] blended = {
                            original[0] * (1 - alpha) + color[0] * alpha,
                            original[1] * (1 - alpha) + color[1] * alpha,
                            original[2] * (1 - alpha) + color[2] * alpha
                    };
                    result.put(i, j, blended);
                }
            }
        }

        // 显示结果
        HighGui.imshow("Segmentation Result", result);
        HighGui.waitKey(0);
    }

    public void showPoints(float[][] points, float[] labels, Mat image) {
        Mat result = image.clone();

        for (int i = 0; i < points.length; i++) {
            Scalar color = labels[i] == 1 ? new Scalar(0, 255, 0) : new Scalar(0, 0, 255);
            Point pt = new Point(points[i][0], points[i][1]);
            Imgproc.circle(result, pt, 10, color, -1);
            Imgproc.circle(result, pt, 10, new Scalar(255, 255, 255), 2);
        }

        HighGui.imshow("Points", result);
        HighGui.waitKey(0);
    }

    public void showBox(float[] box, Mat image) {
        Mat result = image.clone();
        Point pt1 = new Point(box[0], box[1]);
        Point pt2 = new Point(box[2], box[3]);
        Imgproc.rectangle(result, pt1, pt2, new Scalar(0, 255, 0), 2);

        HighGui.imshow("Box", result);
        HighGui.waitKey(0);
    }

    public Mat infer(float[] box, float[][] points, float[] labels, Mat image) throws OrtException {
        // 预处理图像
        PreprocessResult preprocessed = preprocessForEncoder(image);

        // 准备提示点
        float[][] inputBox = {{box[0], box[1]}, {box[2], box[3]}};
        float[] boxLabels = {2, 3};

        float[][] onnxCoord;
        float[] onnxLabel;

        if (points == null || points.length == 0) {
            onnxCoord = inputBox;
            onnxLabel = boxLabels;
        } else {
            onnxCoord = new float[points.length + 2][2];
            onnxLabel = new float[points.length + 2];

            System.arraycopy(points, 0, onnxCoord, 0, points.length);
            System.arraycopy(labels, 0, onnxLabel, 0, labels.length);

            onnxCoord[points.length] = inputBox[0];
            onnxCoord[points.length + 1] = inputBox[1];
            onnxLabel[points.length] = 2;
            onnxLabel[points.length + 1] = 3;
        }

        // 转换坐标
        float[][] transformedCoords = transformCoords(onnxCoord,
                new Size(image.cols(), image.rows()), preprocessed.scale);

        // 准备ONNX输入
        Map<String, OnnxTensor> inputs = new HashMap<>();

        if (needEmbeddings) {
            // 运行编码器获取embeddings
            OrtSession.Result encoderResult = encoderSession.run(
                    Collections.singletonMap(encoderSession.getInputNames().iterator().next(),
                            preprocessed.tensor));
            OnnxValue embeddingValue = encoderResult.get(0);
            inputs.put("image_embeddings", (OnnxTensor) embeddingValue);
        } else {
            inputs.put(rawInputName, preprocessed.tensor);
        }

        // 添加点坐标和标签
        float[][][] coordInput = {transformedCoords};
        float[][] labelInput = {onnxLabel};

        long[] coordShape = {1, (long)transformedCoords.length, 2};
        long[] labelShape = {1, (long)onnxLabel.length};

        FloatBuffer coordBuffer = FloatBuffer.allocate(transformedCoords.length * 2);
        for (float[] coord : transformedCoords) {
            coordBuffer.put(coord);
        }
        coordBuffer.rewind();

        FloatBuffer labelBuffer = FloatBuffer.allocate(onnxLabel.length);
        labelBuffer.put(onnxLabel);
        labelBuffer.rewind();

        inputs.put("point_coords", OnnxTensor.createTensor(env, coordBuffer, coordShape));
        inputs.put("point_labels", OnnxTensor.createTensor(env, labelBuffer, labelShape));

        // 添加掩码输入
        float[][][][] maskInput = new float[1][1][256][256];
        long[] maskShape = {1, 1, 256, 256};
        FloatBuffer maskBuffer = FloatBuffer.allocate(1 * 1 * 256 * 256);
        inputs.put("mask_input", OnnxTensor.createTensor(env, maskBuffer, maskShape));

        // 添加has_mask_input
        float[] hasMaskInput = {0};
        long[] hasMaskShape = {1};
        inputs.put("has_mask_input", OnnxTensor.createTensor(env,
                FloatBuffer.wrap(hasMaskInput), hasMaskShape));

        // 添加原始图像尺寸
        float[] origSize = {image.rows(), image.cols()};
        long[] origSizeShape = {2};
        inputs.put("orig_im_size", OnnxTensor.createTensor(env,
                FloatBuffer.wrap(origSize), origSizeShape));

        // 运行推理
        OrtSession.Result results = session.run(inputs);
        OnnxValue maskValue = results.get(0);
        // 修复mask处理逻辑
        Mat mask = processMaskOutput(maskValue, image.size());

        return mask;
    }

    private Mat processMaskOutput(OnnxValue maskValue, Size originalSize) throws OrtException {
        // 获取mask数据
        Object value = maskValue.getValue();
        Mat mask = new Mat();

        if (value instanceof float[][][][]) {
            float[][][][] maskData = (float[][][][]) value;
            int batchSize = maskData.length;
            int channels = maskData[0].length;
            int height = maskData[0][0].length;
            int width = maskData[0][0][0].length;

            // 通常我们取第一个batch和第一个channel
            mask = new Mat(height, width, CvType.CV_32F);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    mask.put(i, j, maskData[0][0][i][j]);
                }
            }

        } else if (value instanceof float[][][]) {
            float[][][] maskData = (float[][][]) value;
            int batchSize = maskData.length;
            int height = maskData[0].length;
            int width = maskData[0][0].length;

            mask = new Mat(height, width, CvType.CV_32F);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    mask.put(i, j, maskData[0][i][j]);
                }
            }
        } else {
            throw new RuntimeException("Unknown mask output format");
        }

        // 应用阈值
        float maskThreshold = 0.0f;
        Mat binaryMask = new Mat();
        Core.compare(mask, new Scalar(maskThreshold), binaryMask, Core.CMP_GT);

        // 转换为8位
        Mat mask8u = new Mat();
        binaryMask.convertTo(mask8u, CvType.CV_8U, 255);

        // 调整到原始图像尺寸
        Mat resizedMask = new Mat();
        Imgproc.resize(mask8u, resizedMask, originalSize, 0, 0, Imgproc.INTER_NEAREST);

        return resizedMask;
    }

    public void close() throws OrtException {
        if (session != null) session.close();
        if (encoderSession != null) encoderSession.close();
        if (env != null) env.close();
    }
}