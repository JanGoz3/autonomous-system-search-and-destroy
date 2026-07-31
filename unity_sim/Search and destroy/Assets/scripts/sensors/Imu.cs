using UnityEngine;

public class Imu : MonoBehaviour
{
    [Header("Car Rigidbody")]
    public Rigidbody carRigidbody;

    [Header("MPU9255 params")]
    public float accelNoiseStdDev = 0.01f; 
    public float gyroNoiseStdDev = 0.08f;   
    public Vector3 gyroBias = new Vector3(0.70f, 0.82f, 1.50f);

    [Header("Timing")]
    public float timingBudget = 0.025f;

    private Rigidbody rb;
    private float lastUpdateTime = -1f;
    private Vector3 lastVelocity;

    [Header("Preview")]


    public Vector3 cachedAccelerometer = Vector3.zero;
    public Vector3 cachedGyroscope = Vector3.zero;

    void Start()
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
            UpdateSensorHardware();
            lastUpdateTime = currentTime;
        }
    }

    private void UpdateSensorHardware()
    {
        Vector3 currentVelocity = rb.GetPointVelocity(transform.position);
        Vector3 acceleration = (currentVelocity - lastVelocity) / timingBudget;
        lastVelocity = currentVelocity;
        
        acceleration -= Physics.gravity; 
        
        Vector3 localAccel = transform.InverseTransformDirection(acceleration);
        Vector3 localGyro = transform.InverseTransformDirection(rb.angularVelocity) * Mathf.Rad2Deg;

        float imuAccX = -localAccel.x;
        float imuAccY = -localAccel.z;
        float imuAccZ = localAccel.y;

        float imuGyroX = -localGyro.x;
        float imuGyroY = -localGyro.z;
        float imuGyroZ = localGyro.y;

        cachedAccelerometer = new Vector3(
            (imuAccX / 9.81f) + GenerateGaussianNoise(0, accelNoiseStdDev),
            (imuAccY / 9.81f) + GenerateGaussianNoise(0, accelNoiseStdDev),
            (imuAccZ / 9.81f) + GenerateGaussianNoise(0, accelNoiseStdDev)
        );

        cachedGyroscope = new Vector3(
            imuGyroX + gyroBias.x + GenerateGaussianNoise(0, gyroNoiseStdDev),
            imuGyroY + gyroBias.y + GenerateGaussianNoise(0, gyroNoiseStdDev),
            imuGyroZ + gyroBias.z + GenerateGaussianNoise(0, gyroNoiseStdDev)
        );
    }

    private float GenerateGaussianNoise(float mean, float stdDev)
    {
        float u1 = 1.0f - Random.value;
        float u2 = 1.0f - Random.value;
        float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);
        return mean + stdDev * randStdNormal;
    }

    void FixedUpdate()
    {
        TryUpdateHardware();
    }
}