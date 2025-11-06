package com.xulihang;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {
    public static void main(String[] args) {
        try {
            // 模型路径
            String decoderModelPath = "decoder.onnx";
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

            // 示例1: 使用边界框进行推理
            float[] box1 = {380, 230, 480, 520};
            float[][] points1 = {}; // 空点
            float[] labels1 = {};

            System.out.println("运行第一个推理...");
            Mat mask1 = sam.infer(box1, points1, labels1, imageRGB);
            Imgcodecs.imwrite("out.png",mask1);
            //sam.showMask(mask1, imageRGB);
            //sam.showBox(box1, imageRGB);



            // 清理资源
            sam.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}