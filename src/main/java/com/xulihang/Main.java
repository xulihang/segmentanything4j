package com.xulihang;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {
    public static void main(String[] args) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            // 模型路径 - 使用与Python版本相同的模型文件
            String decoderModelPath = "decoder.quant.onnx";
            String encoderModelPath = "encoder.onnx"; // 如果需要

            // 图像路径
            String imagePath = "picture2.jpg";

            // 创建推理器
            SAMOnnxInference sam = new SAMOnnxInference(decoderModelPath, encoderModelPath);

            // 读取图像
            Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("无法加载图像: " + imagePath);
                return;
            }

            // 转换颜色空间 BGR -> RGB
            Mat imageRGB = new Mat();
            Imgproc.cvtColor(image, imageRGB, Imgproc.COLOR_BGR2RGB);

            // 示例1: 使用与Python版本相同的边界框进行推理
            float[] box1 = {210, 200, 350, 500};
            float[][] points1 = {}; // 空点
            float[] labels1 = {};

            //System.out.println("运行第一个推理...");
            Mat mask1 = sam.infer(box1, points1, labels1, imageRGB);
            Imgcodecs.imwrite("out1.png", mask1);
            //System.out.println("第一个掩码已保存为 out1.png");

            // 示例2: 使用与Python版本相同的第二个边界框进行推理
            float[] box2 = {380, 230, 480, 520};
            float[][] points2 = {}; // 空点
            float[] labels2 = {};

            //System.out.println("运行第二个推理...");
            Mat mask2 = sam.infer(box2, points2, labels2, imageRGB);
            Imgcodecs.imwrite("out2.png", mask2);
            //System.out.println("第二个掩码已保存为 out2.png");

            // 清理资源
            sam.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}