using UnityEngine;

public class Imu : MonoBehaviour
{
    [Header("Car Rigidbody")]
    public Rigidbody carRigidbody;

    [Header("MPU9255 params")]
    public float accelNoiseStdDev = 0.005f; 
    public float accelSpeedNoiseMultiplier = 0.15f;
    public float gyroNoiseStdDev = 0.15f;   
    public Vector3 gyroBias = new Vector3(0.79f, 0.73f, 1.46f);
    public Vector3 accelBias = new Vector3(0f, 0f, 0.04f);

    [Header("Timing")]
    public float timingBudget = 0.025f;

    private Rigidbody rb;
    private float lastUpdateTime = -1f;
    private Vector3 lastVelocity;

    [Header("Preview")]


    public Vector3 cachedAccelerometer = Vector3.zero;
    public Vector3 cachedGyroscope = Vector3.zero;

    void Awake()
    {
        rb = carRigidbody;
        lastVelocity = rb.GetPointVelocity(transform.position);
    }

    public Vector3 GetAccelerometer()
    {
        TryUpdateHardware();
        return cachedAccelerometer;
    }

    public Vector3 GetGyroscope()
    {
        TryUpdateHardware();
        return cachedGyroscope;
    }

    private void TryUpdateHardware()
    {
        float currentTime = Time.fixedTime; 
        if (currentTime - lastUpdateTime >= timingBudget)
        {
            float dt = (lastUpdateTime < 0f) ? timingBudget : (currentTime - lastUpdateTime);
            UpdateSensorHardware(dt);
            lastUpdateTime = currentTime;
        }
    }

    private void UpdateSensorHardware(float dt)
    {
        Vector3 currentVelocity = rb.GetPointVelocity(transform.position);
        Vector3 acceleration = (currentVelocity - lastVelocity) / dt;
        lastVelocity = currentVelocity;
        
        acceleration -= Physics.gravity; 
        
        Vector3 localAccel = transform.InverseTransformDirection(acceleration);
        Vector3 localGyro = transform.InverseTransformDirection(rb.angularVelocity) * Mathf.Rad2Deg;

        float imuAccX = -localAccel.x;
        float imuAccY = -localAccel.z;
        float imuAccZ = localAccel.y;

        float imuGyroX = -localGyro.x;
        float imuGyroY = -localGyro.z;
        float imuGyroZ = -localGyro.y;

        float currentSpeed = currentVelocity.magnitude;
        float dynamicAccelNoise = accelNoiseStdDev + (currentSpeed * accelSpeedNoiseMultiplier);

        cachedAccelerometer = new Vector3(
            (imuAccX / 9.81f) + accelBias.x + NoiseGenerator.GenerateGaussian(0, dynamicAccelNoise),
            (imuAccY / 9.81f) + accelBias.y + NoiseGenerator.GenerateGaussian(0, dynamicAccelNoise),
            (imuAccZ / 9.81f) + accelBias.z + NoiseGenerator.GenerateGaussian(0, dynamicAccelNoise)
        );

        cachedGyroscope = new Vector3(
            imuGyroX + gyroBias.x + NoiseGenerator.GenerateGaussian(0, gyroNoiseStdDev),
            imuGyroY + gyroBias.y + NoiseGenerator.GenerateGaussian(0, gyroNoiseStdDev),
            imuGyroZ + gyroBias.z + NoiseGenerator.GenerateGaussian(0, gyroNoiseStdDev)
        );
    }

    void FixedUpdate()
    {
        TryUpdateHardware();
    }
}