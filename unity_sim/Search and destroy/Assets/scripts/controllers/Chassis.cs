using UnityEngine;

public class Chassis : MonoBehaviour
{
    [Header("Sub-Components")]
    public Motor motor;
    public Steering steering;
    public ServosCamera servosCamera;

    [Header("Hardware References")]
    public Rigidbody carRigidbody;
    public WheelCollider frontLeftWheel, frontRightWheel, rearLeftWheel, rearRightWheel;

    [Header("Sensors")]
    public Imu imu;
    public TofSensor tofSensor;
    public YoloVision yoloVision;

    void Awake() 
    {
        Initialize();
    }

    public void Initialize()
    {
        WheelCollider[] allWheels = { frontLeftWheel, frontRightWheel, rearLeftWheel, rearRightWheel };
        
        motor.Initialize(allWheels, carRigidbody);
        steering.Initialize(frontLeftWheel, frontRightWheel);
        servosCamera.Initialize();
    }

    public void SetSpeed(float speed)
    {
        motor.SetSpeed(speed);
    }

    public void SetSteering(float swing)
    {
        steering.SetSteering(swing);
    }

    public void SetCameraServos(float pitch, float yaw)
    {
        servosCamera.SetPitchYaw(pitch, yaw);
    }

    public void SetNeutral()
    {
        motor.StopMotor();
        steering.SetNeutralSwing();
        servosCamera.SetNeutralSwing();
    }

    public float[] GetTelemetryState()
    {
        Vector3 accel = imu.GetAccelerometer();
        Vector3 gyro = imu.GetGyroscope();
        var cameraState = servosCamera.GetCurrentPitchYaw();

        float maxAccel = 16f;
        float maxGyro = 2000f;
        float maxTof = 4000f;

        float[] yoloData = yoloVision.GetYoloState();
        // if (yoloData != null && yoloData.Length > 0) {
        //     Debug.Log("Yolo data: " + string.Join(", ", yoloData));
        // } else {
        //     Debug.Log("Yolo data: No objects detected.");
        // }

        bool yoloDetected = yoloData != null && yoloData.Length >= 6;

        float[] telemetry = new float[]
        {
            motor.GetCurrentSetSpeed(),
            steering.GetCurrentSetSwing(),
            cameraState.pitch,
            cameraState.yaw,
            
            Mathf.Clamp(accel.x / maxAccel, -1f, 1f),
            Mathf.Clamp(accel.y / maxAccel, -1f, 1f),
            Mathf.Clamp(accel.z / maxAccel, -1f, 1f),
            
            Mathf.Clamp(gyro.x / maxGyro, -1f, 1f),
            Mathf.Clamp(gyro.y / maxGyro, -1f, 1f),
            Mathf.Clamp(gyro.z / maxGyro, -1f, 1f),
            
            Mathf.Clamp(tofSensor.GetDistance() / maxTof, 0f, 1f),

            yoloDetected ? yoloData[0] : 0f, // Bounding Box Center X
            yoloDetected ? yoloData[1] : 0f, // Bounding Box Center Y
            yoloDetected ? yoloData[2] : 0f, // Bounding Box Width
            yoloDetected ? yoloData[3] : 0f, // Bounding Box Height
            yoloDetected ? yoloData[4] : 0f, // YOLO Confidence Score
            yoloDetected ? yoloData[5] : -1f  // YOLO Class ID
        };
        
        return telemetry;
    }
}