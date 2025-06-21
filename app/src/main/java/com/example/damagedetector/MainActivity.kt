package com.example.damagedetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.*
import android.util.Size
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var btnDetect: Button
    private lateinit var txtResult: TextView
    private lateinit var previewSurface: SurfaceView

    private val CAMERA_PERMISSION_REQUEST_CODE = 100
    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private lateinit var previewRequestBuilder: CaptureRequest.Builder
    private lateinit var imageReader: ImageReader
    private lateinit var backgroundHandler: Handler
    private lateinit var backgroundThread: HandlerThread

    private lateinit var tflite: Interpreter
    private var isDetecting = false

    private val MODEL_INPUT_SIZE = 224
    private var previewSize: Size = Size(640, 480)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewSurface = findViewById(R.id.camera_preview)
        btnDetect = findViewById(R.id.btn_detect)
        txtResult = findViewById(R.id.txt_all_results)

        loadModel()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
        } else {
            startCamera()
        }

        btnDetect.setOnClickListener {
            isDetecting = true
            txtResult.text = "Detecting..."
            Toast.makeText(this, "Detection started", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startCamera() {
        startBackgroundThread()

        previewSurface.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                openCamera()
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}
            override fun surfaceDestroyed(holder: SurfaceHolder) {}
        })
    }

    private fun openCamera() {
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        val cameraId = cameraManager.cameraIdList[0]

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return

        setupImageReader(cameraId)

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createCameraPreviewSession()
            }
            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                cameraDevice = null
            }
            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                cameraDevice = null
            }
        }, backgroundHandler)
    }

    private fun setupImageReader(cameraId: String) {
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        val streamConfigMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val outputSizes = streamConfigMap?.getOutputSizes(SurfaceHolder::class.java)
        previewSize = outputSizes?.maxByOrNull { it.width * it.height } ?: Size(640, 480)

        imageReader = ImageReader.newInstance(previewSize.width, previewSize.height, ImageFormat.YUV_420_888, 2)
        imageReader.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)
    }

    private fun createCameraPreviewSession() {
        try {
            val previewSurfaceHolder = previewSurface.holder.surface
            val imageReaderSurface = imageReader.surface

            previewRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            previewRequestBuilder.addTarget(previewSurfaceHolder)
            previewRequestBuilder.addTarget(imageReaderSurface)

            cameraDevice!!.createCaptureSession(listOf(previewSurfaceHolder, imageReaderSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        previewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                        captureSession!!.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler)
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(this@MainActivity, "Camera configuration failed", Toast.LENGTH_SHORT).show()
                    }
                }, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private val imageAvailableListener = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener

        if (isDetecting) {
            val bitmap = imageToBitmap(image)
            val input = preprocess(bitmap)
            val output = runInference(input)

            runOnUiThread {
                txtResult.text = parseResult(output)
            }
        }
        image.close()
    }

    private fun imageToBitmap(image: Image): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val jpegBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        val byteBuffer = ByteBuffer.allocateDirect(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3)  // Quantized = 1 byte per pixel channel
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        for (pixel in intValues) {
            byteBuffer.put(((pixel shr 16) and 0xFF).toByte())  // R
            byteBuffer.put(((pixel shr 8) and 0xFF).toByte())   // G
            byteBuffer.put((pixel and 0xFF).toByte())           // B
        }

        return byteBuffer
    }



    private fun runInference(input: ByteBuffer): FloatArray {
        val output = Array(1) { ByteArray(3) }  // Only 3 classes now
        tflite.run(input, output)

        val probabilities = FloatArray(3)
        for (i in output[0].indices) {
            probabilities[i] = (output[0][i].toUByte().toFloat()) / 255.0f
        }
        return probabilities
    }



    private fun parseResult(output: FloatArray): String {
        val labels = arrayOf("No Damage", "Scratch", "Dent")

        val sb = StringBuilder()
        for (i in labels.indices) {
            sb.append("${labels[i]}: ${(output[i] * 100).toInt()}%\n")
        }
        return sb.toString()
    }




    private fun loadModel() {
        val modelBuffer = loadModelFile()
        tflite = Interpreter(modelBuffer)
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread.looper)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopBackgroundThread()
    }

    private fun stopBackgroundThread() {
        backgroundThread.quitSafely()
        backgroundThread.join()
    }
}
