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

            yoloData[0], // Bounding Box Center X
            yoloData[1], // Bounding Box Center Y
            yoloData[2], // Bounding Box Width
            yoloData[3], // Bounding Box Height
            yoloData[4], // YOLO Confidence Score
            yoloData[5]  // YOLO Class ID
        };
        
        return telemetry;
    }
}