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

    private float[] m_Telemetry = new float[17];

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

        m_Telemetry[0] = motor.GetCurrentSetSpeed();
        m_Telemetry[1] = steering.GetCurrentSetSwing();
        m_Telemetry[2] = cameraState.pitch;
        m_Telemetry[3] = cameraState.yaw;
        
        m_Telemetry[4] = Mathf.Clamp(accel.x / maxAccel, -1f, 1f);
        m_Telemetry[5] = Mathf.Clamp(accel.y / maxAccel, -1f, 1f);
        m_Telemetry[6] = Mathf.Clamp(accel.z / maxAccel, -1f, 1f);
        
        m_Telemetry[7] = Mathf.Clamp(gyro.x / maxGyro, -1f, 1f);
        m_Telemetry[8] = Mathf.Clamp(gyro.y / maxGyro, -1f, 1f);
        m_Telemetry[9] = Mathf.Clamp(gyro.z / maxGyro, -1f, 1f);
        
        m_Telemetry[10] = Mathf.Clamp(tofSensor.GetDistance() / maxTof, 0f, 1f);

        m_Telemetry[11] = yoloDetected ? yoloData[0] : 0f;
        m_Telemetry[12] = yoloDetected ? yoloData[1] : 0f;
        m_Telemetry[13] = yoloDetected ? yoloData[2] : 0f;
        m_Telemetry[14] = yoloDetected ? yoloData[3] : 0f;
        m_Telemetry[15] = yoloDetected ? yoloData[4] : 0f;
        
        // Remember to mask the Class ID for the Driver network as we discussed!
        m_Telemetry[16] = yoloDetected ? 1f : 0f;
        
        return m_Telemetry;
    }
}