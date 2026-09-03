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

    private const int YoloFeatureCount = 27;
    private readonly float[] m_Telemetry = new float[11 + YoloFeatureCount];

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

        // if (yoloData != null && yoloData.Length > 0) {
        //     Debug.Log("Yolo data: " + string.Join(", ", yoloData));
        // } else {
        //     Debug.Log("Yolo data: No objects detected.");
        // }

        // hardware telemetry
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

        //yolo state
        if (yoloVision != null)
        {
            float[] yoloData = yoloVision.GetYoloState();
            if (yoloData != null && yoloData.Length >= YoloFeatureCount)
            {
                // Fast block copy of all 27 floats into m_Telemetry starting at index 11
                System.Array.Copy(yoloData, 0, m_Telemetry, 11, YoloFeatureCount);
            }
            else
            {
                System.Array.Clear(m_Telemetry, 11, YoloFeatureCount);
            }
        }
        else
        {
            System.Array.Clear(m_Telemetry, 11, YoloFeatureCount);
        }        
        return m_Telemetry;
    }
}