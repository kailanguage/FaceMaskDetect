package cn.kailang.facemaskdetection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraInfoUnavailableException;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.google.mlkit.common.MlKitException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import cn.kailang.facemaskdetection.livepreview.CameraXViewModel;
import cn.kailang.facemaskdetection.livepreview.FaceDetectorProcessor;
import cn.kailang.facemaskdetection.livepreview.GraphicOverlay;
import cn.kailang.facemaskdetection.livepreview.VisionImageProcessor;
import cn.kailang.facemaskdetection.tflite.Detector;
import cn.kailang.facemaskdetection.tflite.FaceMaskDetectionAPIModel;
import cn.kailang.facemaskdetection.utils.BitmapUtils;

import static cn.kailang.facemaskdetection.TFliteConfig.TF_API_INPUT_SIZE;
import static cn.kailang.facemaskdetection.TFliteConfig.TF_API_IS_QUANTIZED;
import static cn.kailang.facemaskdetection.TFliteConfig.TF_API_LABELS_FILE;
import static cn.kailang.facemaskdetection.TFliteConfig.TF_API_MODEL_FILE;

public class CameraXLivePreviewActivity extends AppCompatActivity implements  CompoundButton.OnCheckedChangeListener,ActivityCompat.OnRequestPermissionsResultCallback {
    private static final String TAG = "CameraXLivePreview";
    private static final int PERMISSION_REQUESTS = 1;

    private PreviewView previewView;
    private GraphicOverlay graphicOverlay;
    @Nullable private VisionImageProcessor imageProcessor;

    @Nullable
    private ProcessCameraProvider cameraProvider;
    @Nullable
    private Preview previewUseCase;
    @Nullable
    private ImageAnalysis analysisUseCase;

    private int lensFacing = CameraSelector.LENS_FACING_BACK;
    private CameraSelector cameraSelector;
    private boolean needUpdateGraphicOverlayImageSourceInfo;

    private Detector detector;
    private StringBuilder sbInfo;
    private String ans;
    private TextView resultView;
    private TextView confidenceView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
            Toast.makeText(
                    getApplicationContext(),
                    "CameraX is only supported on SDK version >=21. Current SDK version is "
                            + Build.VERSION.SDK_INT,
                    Toast.LENGTH_LONG)
                    .show();
            return;
        }
        cameraSelector = new CameraSelector.Builder().requireLensFacing(lensFacing).build();
        setContentView(R.layout.activity_camerax_live_preview);
        initView();
        initCameraX();
    }

    @Override
    public void onResume() {
        super.onResume();
        bindAllCameraUseCases();
    }

    private void initView() {
        previewView = findViewById(R.id.preview_view);  
        graphicOverlay = findViewById(R.id.graphic_overlay);
        resultView=findViewById(R.id.result);
        confidenceView=findViewById(R.id.confidence);
        ToggleButton facingSwitch = findViewById(R.id.facing_switch_button);
        facingSwitch.setOnCheckedChangeListener(this);
        if (!allPermissionsGranted()) {
            getRuntimePermissions();
        }
    }

    private void initCameraX() {
        new ViewModelProvider(this, ViewModelProvider.AndroidViewModelFactory.getInstance(getApplication()))
                .get(CameraXViewModel.class)
                .getProcessCameraProvider()
                .observe(
                        this,
                        provider -> {
                            cameraProvider = provider;
                            if (allPermissionsGranted()) {
                                bindAllCameraUseCases();
                            }
                        });
    }

    private void bindAllCameraUseCases() {
        if (cameraProvider != null) {
            // As required by CameraX API, unbinds all use cases before trying to re-bind any of them.
            cameraProvider.unbindAll();
            bindPreviewUseCase();
            bindAnalysisUseCase();
        }
    }

    private void bindPreviewUseCase() {
        if (cameraProvider == null) {
            return;
        }
        if (previewUseCase != null) {
            cameraProvider.unbind(previewUseCase);
        }
        Preview.Builder builder = new Preview.Builder();
//        Size targetResolution = PreferenceUtils.getCameraXTargetResolution(this);
//        if (targetResolution != null) {
//            builder.setTargetResolution(targetResolution);
//        }
        previewUseCase = builder.build();
        previewUseCase.setSurfaceProvider(previewView.getSurfaceProvider());
        cameraProvider.bindToLifecycle(this, cameraSelector, previewUseCase);
    }

    private void initFaceMaskDetect() {
        try {
            detector =
                    FaceMaskDetectionAPIModel.create(
                            this,
                            TF_API_MODEL_FILE,
                            TF_API_LABELS_FILE,
                            TF_API_INPUT_SIZE,
                            TF_API_IS_QUANTIZED);
        } catch (final IOException e) {
            e.printStackTrace();
            Log.e("error", "Exception initializing Detector!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    private void bindAnalysisUseCase() {
        if (cameraProvider == null) {
            return;
        }
        if (analysisUseCase != null) {
            cameraProvider.unbind(analysisUseCase);
        }
        imageProcessor = new FaceDetectorProcessor(this, new FaceDetectorProcessor.FaceInfoCallback() {
            @Override
            public void isHaveFace(boolean isHave) {
                if(isHave) {
                    resultView.setText("Result：" + ans);
                    confidenceView.setText(sbInfo.toString().trim());
                }else {
                    resultView.setText("未检测到人脸！");
                    confidenceView.setText("");
                }
            }
        });

        initFaceMaskDetect();

        ImageAnalysis.Builder builder = new ImageAnalysis.Builder();

        analysisUseCase = builder.build();

        needUpdateGraphicOverlayImageSourceInfo = true;
        analysisUseCase.setAnalyzer(
                // imageProcessor.processImageProxy will use another thread to run the detection underneath,
                // thus we can just runs the analyzer itself on main thread.
                ContextCompat.getMainExecutor(this),
                imageProxy -> {
                    if (needUpdateGraphicOverlayImageSourceInfo) {
                        boolean isImageFlipped = lensFacing == CameraSelector.LENS_FACING_FRONT;
                        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
                        if (rotationDegrees == 0 || rotationDegrees == 180) {
                            graphicOverlay.setImageSourceInfo(
                                    imageProxy.getWidth(), imageProxy.getHeight(), isImageFlipped);
                        } else {
                            graphicOverlay.setImageSourceInfo(
                                    imageProxy.getHeight(), imageProxy.getWidth(), isImageFlipped);
                        }
                        needUpdateGraphicOverlayImageSourceInfo = false;
                    }
                    try {
                        imageProcessor.processImageProxy(imageProxy, graphicOverlay);

                        //face mask detect

                        if (detector != null) {
                            List<Detector.Recognition> recognitions = detector.recognizeImage(BitmapUtils.getBitmap(imageProxy));
                            sbInfo  =new StringBuilder();
                            float cont=0f;
                            for (int i = 0; i < recognitions.size(); i++) {
                                String title=recognitions.get(i).getTitle();
                                float c=recognitions.get(i).getConfidence();
                                sbInfo.append(title+":"+c+"\n");
                                if(c>cont){
                                    cont=c;
                                    ans=title;
                                }
                            }
                        } else {
                            Log.e(TAG, "Null imageProcessor, please check adb logs for imageProcessor creation error");
                        }


                    } catch (MlKitException e) {
                        Log.e(TAG, "Failed to process image. Error: " + e.getLocalizedMessage());
                        Toast.makeText(getApplicationContext(), e.getLocalizedMessage(), Toast.LENGTH_SHORT)
                                .show();
                    }
                });

        cameraProvider.bindToLifecycle(this, cameraSelector, analysisUseCase);
    }



    /******permission********/


    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    this.getPackageManager()
                            .getPackageInfo(this.getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                return false;
            }
        }
        return true;
    }

    private void getRuntimePermissions() {
        List<String> allNeededPermissions = new ArrayList<>();
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                allNeededPermissions.add(permission);
            }
        }

        if (!allNeededPermissions.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this, allNeededPermissions.toArray(new String[0]), PERMISSION_REQUESTS);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        Log.i(TAG, "Permission granted!");
        if (allPermissionsGranted()) {
            bindAllCameraUseCases();
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private static boolean isPermissionGranted(Context context, String permission) {
        if (ContextCompat.checkSelfPermission(context, permission)
                == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission granted: " + permission);
            return true;
        }
        Log.i(TAG, "Permission NOT granted: " + permission);
        return false;
    }

    @Override
    public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
        Log.d(TAG, "Set facing");
        if (cameraProvider == null) {
            return;
        }

        int newLensFacing =
                lensFacing == CameraSelector.LENS_FACING_FRONT
                        ? CameraSelector.LENS_FACING_BACK
                        : CameraSelector.LENS_FACING_FRONT;
        CameraSelector newCameraSelector =
                new CameraSelector.Builder().requireLensFacing(newLensFacing).build();
        try {
            if (cameraProvider.hasCamera(newCameraSelector)) {
                lensFacing = newLensFacing;
                cameraSelector = newCameraSelector;
                bindAllCameraUseCases();
                return;
            }
        } catch (CameraInfoUnavailableException e) {
            // Falls through
        }
        Toast.makeText(
                getApplicationContext(),
                "This device does not have lens with facing: " + newLensFacing,
                Toast.LENGTH_SHORT)
                .show();
    }
}