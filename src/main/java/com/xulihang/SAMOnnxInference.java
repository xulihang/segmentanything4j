package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.List;

public class SAMOnnxInference {
    
    static {
        nu.pattern.OpenCV.loadLocally(); // 加载 OpenCV 本地库
    }
    
    private OrtSession session;
    private OrtSession encoderSession;
    private boolean needEmbeddings;
    private String rawInputName;
    private int targetSize = 1024;
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std = {0.229f, 0.224f, 0.225f};
    
    public SAMOnnxInference(String modelPath, String encoderPath) throws OrtException {
        // 初始化 ONNX Runtime 会话
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        session = env.createSession(modelPath, sessionOptions);
        
        // 检查是否需要图像嵌入
        List<String> inputNames = new ArrayList<String>();
        for (var name:session.getInputNames()) {
            inputNames.add(name);
        }
        needEmbeddings = inputNames.contains("image_embeddings");
        
        if (needEmbeddings) {
            if (encoderPath == null || encoderPath.isEmpty()) {
                throw new IllegalArgumentException("模型需要图像嵌入，但未提供编码器路径");
            }
            encoderSession = env.createSession(encoderPath, sessionOptions);
        } else {
            // 查找原始图像输入名称
            for (String name : inputNames) {
                if (name.equals("image") || name.equals("images") || 
                    name.equals("pixel_values") || name.equals("input_image")) {
                    rawInputName = name;
                    break;
                }
            }
            if (rawInputName == null && !inputNames.isEmpty()) {
                rawInputName = inputNames.get(0);
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
        int h0 = image.rows(), w0 = image.cols();
        float scale = targetSize / (float) Math.max(h0, w0);
        
        int newW = (int) (w0 * scale);
        int newH = (int) (h0 * scale);
        
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_LINEAR);
        
        Mat canvas = Mat.zeros(targetSize, targetSize, CvType.CV_8UC3);
        Mat roi = canvas.submat(0, newH, 0, newW);
        resized.copyTo(roi);
        
        // 转换为 float 并归一化
        Mat imgFloat = new Mat();
        canvas.convertTo(imgFloat, CvType.CV_32F, 1.0 / 255.0);
        
        // 应用归一化
        List<Mat> channels = new ArrayList<>();
        Core.split(imgFloat, channels);
        
        for (int i = 0; i < 3; i++) {
            Core.subtract(channels.get(i), new Scalar(mean[i]), channels.get(i));
            Core.divide(channels.get(i), new Scalar(std[i]), channels.get(i));
        }
        
        Mat normalized = new Mat();
        Core.merge(channels, normalized);
        
        // HWC -> CHW
        Mat chw = new Mat();
        Core.transpose(normalized, chw);
        Mat finalTensorMat = chw.reshape(1, new int[]{1, 3, targetSize, targetSize});
        
        // 创建 OnnxTensor
        float[] tensorData = new float[3 * targetSize * targetSize];
        finalTensorMat.get(0, 0, tensorData);
        
        long[] shape = {1, 3, targetSize, targetSize};
        OnnxTensor tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), 
            FloatBuffer.wrap(tensorData), shape);
        
        return new PreprocessResult(tensor, scale, new Size(newW, newH));
    }
    
    public float[] transformCoords(float[][] coords, Size origSize, float scale) {
        float[] transformed = new float[coords.length * 2];
        for (int i = 0; i < coords.length; i++) {
            transformed[i * 2] = coords[i][0] * scale;
            transformed[i * 2 + 1] = coords[i][1] * scale;
        }
        return transformed;
    }
    
    public Mat infer(float[] box, float[][] points, float[] labels, Mat image) throws OrtException {
        // 准备输入数据
        float[][] inputBox = {{box[0], box[1]}, {box[2], box[3]}}; // [[x0,y0],[x1,y1]]
        float[] boxLabels = {2.0f, 3.0f};
        
        // 合并点和框坐标
        List<float[]> allCoords = new ArrayList<>();
        List<Float> allLabels = new ArrayList<>();
        
        // 添加点
        for (int i = 0; i < points.length; i++) {
            allCoords.add(points[i]);
            allLabels.add(labels[i]);
        }
        
        // 添加框角点
        allCoords.add(inputBox[0]);
        allCoords.add(inputBox[1]);
        allLabels.add(2.0f);
        allLabels.add(3.0f);
        
        float[][] onnxCoord = allCoords.toArray(new float[0][]);
        float[] onnxLabel = new float[allLabels.size()];
        for (int i = 0; i < allLabels.size(); i++) {
            onnxLabel[i] = allLabels.get(i);
        }
        
        // 变换坐标
        float scale;
        OnnxTensor imageTensor = null;
        OnnxTensor embeddingsTensor = null;
        
        if (needEmbeddings) {
            PreprocessResult result = preprocessForEncoder(image);
            imageTensor = result.tensor;
            scale = result.scale;
            
            // 运行编码器
            Map<String, OnnxTensor> encoderInputs = new HashMap<>();
            encoderInputs.put(encoderSession.getInputNames().iterator().next(), imageTensor);
            OrtSession.Result encoderOutputs = encoderSession.run(encoderInputs);
            
            OnnxTensor embeddings = (OnnxTensor) encoderOutputs.get(0);
            embeddingsTensor = embeddings;
        } else {
            PreprocessResult result = preprocessForEncoder(image);
            imageTensor = result.tensor;
            scale = result.scale;
        }
        
        float[] transformedCoords = transformCoords(onnxCoord, 
            new Size(image.cols(), image.rows()), scale);
        
        // 准备其他输入
        float[][][][] maskInput = new float[1][1][256][256]; // 全零
        float[] hasMaskInput = {0.0f};
        float[] origImSize = {(float) image.rows(), (float) image.cols()};
        
        // 构建输入映射
        Map<String, OnnxTensor> inputs = new HashMap<>();
        
        for (String inputName : session.getInputNames()) {
            if (inputName.equals("image_embeddings") && needEmbeddings) {
                inputs.put(inputName, embeddingsTensor);
            } else if (inputName.equals("point_coords") || 
                       inputName.equals("point_coordinates") || 
                       inputName.equals("point_coords_coord")) {
                long[] coordShape = {1, onnxCoord.length, 2};
                inputs.put(inputName, OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                    FloatBuffer.wrap(transformedCoords), coordShape));
            } else if (inputName.equals("point_labels") || inputName.equals("point_label")) {
                long[] labelShape = {1, onnxLabel.length};
                inputs.put(inputName, OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                    FloatBuffer.wrap(onnxLabel), labelShape));
            } else if (inputName.equals("mask_input")) {
                inputs.put(inputName, OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                    maskInput));
            } else if (inputName.equals("has_mask_input")) {
                inputs.put(inputName, OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                    hasMaskInput));
            } else if (inputName.equals("orig_im_size") || 
                       inputName.equals("original_size") || 
                       inputName.equals("orig_size")) {
                inputs.put(inputName, OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                    origImSize));
            } else if (!needEmbeddings && inputName.equals(rawInputName)) {
                inputs.put(inputName, imageTensor);
            }
        }
        
        // 运行推理
        OrtSession.Result results = session.run(inputs);
        OnnxTensor masksTensor = (OnnxTensor) results.get(0);
        float[][][][] masks = (float[][][][]) masksTensor.getValue();

        // 应用阈值 - 修复后的代码
        float maskThreshold = 0.0f;
        int batchSize = masks.length;
        int numMasks = masks[0].length;
        int height = masks[0][0].length;
        int width = masks[0][0][0].length;

        boolean[][][][] binaryMasks = new boolean[batchSize][numMasks][height][width];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < numMasks; j++) {
                for (int k = 0; k < height; k++) {
                    for (int l = 0; l < width; l++) {
                        binaryMasks[i][j][k][l] = masks[i][j][k][l] > maskThreshold;
                    }
                }
            }
        }

        // 转换为 OpenCV Mat - 修复后的代码
        // 选择第一个批次和第一个掩码
        if (batchSize > 0 && numMasks > 0) {
            boolean[][] mask = binaryMasks[0][0];
            Mat maskMat = new Mat(height, width, CvType.CV_8UC1);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    maskMat.put(i, j, mask[i][j] ? 255 : 0);
                }
            }

            // 调整到原始图像大小
            Mat maskResized = new Mat();
            Imgproc.resize(maskMat, maskResized, new Size(image.cols(), image.rows()),
                    0, 0, Imgproc.INTER_NEAREST);

            // 清理资源
            if (imageTensor != null) imageTensor.close();
            if (embeddingsTensor != null) embeddingsTensor.close();
            masksTensor.close();

            return maskResized;
        } else {
            throw new RuntimeException("没有生成有效的掩码");
        }
    }
    
    public static Mat loadImage(String path) {
        Mat image = Imgcodecs.imread(path);
        if (image.empty()) {
            throw new RuntimeException("无法加载图像: " + path);
        }
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);
        return image;
    }
    
    public static void showResult(Mat image, Mat mask, float[] box, float[][] points, float[] labels) {
        // 创建带掩码的可视化图像
        Mat visualization = image.clone();
        
        // 应用掩码颜色
        Mat colorMask = new Mat(mask.size(), CvType.CV_8UC3, new Scalar(30, 144, 255));
        Core.bitwise_and(colorMask, colorMask, colorMask, mask);
        
        // 将掩码叠加到原图
        Mat maskedRegion = new Mat();
        Core.addWeighted(visualization, 1.0, colorMask, 0.6, 0.0, maskedRegion);
        colorMask.copyTo(maskedRegion, mask);
        
        // 绘制边界框
        Imgproc.rectangle(maskedRegion, 
            new Point(box[0], box[1]),
            new Point(box[2], box[3]), 
            new Scalar(0, 255, 0), 2);
        
        // 绘制点
        for (int i = 0; i < points.length; i++) {
            Scalar color = labels[i] == 1 ? new Scalar(0, 255, 0) : new Scalar(255, 0, 0);
            Imgproc.circle(maskedRegion, 
                new Point(points[i][0], points[i][1]), 
                5, color, -1);
        }
        
        // 显示图像
        Imgcodecs.imwrite("mask.png",maskedRegion);
    }
    

    
    public static void main(String[] args) {
        try {
            String modelPath = "decoder.onnx";
            String encoderPath = "encoder.onnx";
            String imagePath = "picture2.jpg";
            
            SAMOnnxInference sam = new SAMOnnxInference(modelPath, encoderPath);
            Mat image = loadImage(imagePath);
            
            // 示例推理：框 [210, 200, 350, 500]，无点
            float[] box = {210, 200, 350, 500};
            float[][] points = {}; // 空点数组
            float[] labels = {};   // 空标签数组
            
            Mat mask = sam.infer(box, points, labels, image);
            Imgcodecs.imwrite("mask.png",mask);
            //showResult(image, mask, box, points, labels);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}