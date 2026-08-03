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

    void Start()
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

        float[] telemetry = new float[]
        {
            motor.GetCurrentSetSpeed(),
            steering.GetCurrentSetSwing(),
            cameraState.pitch,
            cameraState.yaw,
            accel.x,
            accel.y,
            accel.z,
            gyro.x,
            gyro.y,
            gyro.z,
            tofSensor.GetDistance()
        };

        return telemetry;
    }
}