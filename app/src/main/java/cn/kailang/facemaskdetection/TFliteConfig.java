package cn.kailang.facemaskdetection;

public class TFliteConfig {
    // Configuration values for the tflite model.
    public static final int TF_API_INPUT_SIZE = 200;//模型输入尺寸的大小
    public static final boolean TF_API_IS_QUANTIZED = false;//是否为量化模型
    public static final String TF_API_MODEL_FILE = "face_mask_detect_notf.tflite";
    public static final String TF_API_LABELS_FILE = "face_mask_labels.txt";
    public static final float MINIMUM_CONFIDENCE_TF_API = 0.5f;
}
