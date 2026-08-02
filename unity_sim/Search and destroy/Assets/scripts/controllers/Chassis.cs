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
    public Transform frontLeftMesh, frontRightMesh, rearLeftMesh, rearRightMesh;

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
}