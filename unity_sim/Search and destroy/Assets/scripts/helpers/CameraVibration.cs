using System;
using UnityEngine;

public class CameraVibration : MonoBehaviour
{
    [Header("Vibration settings")]
    public float positionalAmplitude = 0.01f;
    public float rotationalAmplitude = 1.5f;
    public float frequency = 25.0f;
    public Rigidbody carRb;
    public Motor carMotor;
    [Range(0f, 1f)]
    public float stationaryVibrationMultiplier = 0.2f;
    private Vector3 initialLocalPosition;
    private Quaternion initialLocalRotation;
    
    void Start()
    {
        initialLocalPosition = transform.localPosition;
        initialLocalRotation = transform.localRotation;
    }

    
    void FixedUpdate()
    {
        float intensity = 1f;
        if (carRb != null && carMotor != null)
        {
            float speedRatio = Mathf.Clamp01(carRb.linearVelocity.magnitude / carMotor.maxSpeedMetersPerSec);
            intensity = Mathf.Lerp(stationaryVibrationMultiplier, 1.0f, speedRatio);
        }
        float time = Time.time * frequency;
        
        // Calculate continuous noise (-0.5 centers the wave around 0)
        float noiseX = Mathf.PerlinNoise(time, 0f) - 0.5f;
        float noiseY = Mathf.PerlinNoise(0f, time) - 0.5f;
        float noiseZ = Mathf.PerlinNoise(time, time) - 0.5f;

        //Apply positional vibration
        Vector3 posOffset = new Vector3(noiseX, noiseY, noiseZ) * (positionalAmplitude * intensity);
        transform.localPosition = initialLocalPosition + posOffset;

        float rotNoiseX = Mathf.PerlinNoise(time + 10f, 0f) - 0.5f;
        float rotNoiseY = Mathf.PerlinNoise(0f, time + 10f) - 0.5f;
        float rotNoiseZ = Mathf.PerlinNoise(time + 10f, time + 10f) - 0.5f;

        Vector3 rotOffset = new Vector3(rotNoiseX, rotNoiseY, rotNoiseZ) * (rotationalAmplitude * intensity);
        transform.localRotation = initialLocalRotation * Quaternion.Euler(rotOffset);

    }
}
