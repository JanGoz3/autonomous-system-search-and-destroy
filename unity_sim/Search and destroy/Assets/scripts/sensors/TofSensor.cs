using UnityEngine;

public class TofSensor : MonoBehaviour
{
    [Header("Pololu VL53L1X params")]
    public ushort maxDistance = 4000;
    public float noiseStdDev = 5f;

    [Header("Timing")]
    public float timingBudget = 0.05f; 

    private float lastUpdateTime = -1f;
    private ushort cachedDistance = 4000;

    public ushort GetDistance()
    {
        float currentTime = Time.time;
        if (currentTime - lastUpdateTime >= timingBudget)
        {
            UpdateSensorHardware();
            lastUpdateTime = currentTime;
        }

        return cachedDistance;
    }

    private void UpdateSensorHardware()
    {
        float maxDistanceMeters = maxDistance / 1000f;

        if (Physics.Raycast(transform.position, transform.forward, out RaycastHit hit, maxDistanceMeters))
        {
            float distanceMm = hit.distance * 1000f;
            
            float noiseMm = NoiseGenerator.GenerateGaussian(0f, noiseStdDev);
            
            float measuredMm = Mathf.Clamp(distanceMm + noiseMm, 0f, maxDistance);
            cachedDistance = (ushort)measuredMm;
            
            Debug.DrawLine(transform.position, hit.point, Color.red, timingBudget);
        }
        else
        {
            cachedDistance = maxDistance;
            Debug.DrawRay(transform.position, transform.forward * maxDistanceMeters, Color.green, timingBudget);
        }
    }

    void Update()
    {
        GetDistance();
    }
}