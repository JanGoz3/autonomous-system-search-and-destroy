using UnityEngine;

public class Motor : MonoBehaviour
{
    [Header("Motor Setup (Traxxas Titan 12T 550 & XL-5 ESC)")]
    public float maxMotorTorque = 150f;
    public float maxBrakeTorque = 300f;
    public float maxSpeedMetersPerSec = 13.41f; 

    private WheelCollider[] m_wheels;
    private Rigidbody m_rb;
    private bool m_isInitialized = false;

    private float m_currentSetSpeed = 0.0f;

    public void Initialize(WheelCollider[] wheels, Rigidbody rb)
    {
        m_wheels = wheels;
        m_rb = rb;
        m_isInitialized = true;
    }

    public void SetSpeed(float speed)
    {
        if (!m_isInitialized) return;

        speed = Mathf.Clamp(speed, -1f, 1f);
        m_currentSetSpeed = speed;
        
        float localVelocityZ = transform.InverseTransformDirection(m_rb.linearVelocity).z;
        float currentSpeedMPS = m_rb.linearVelocity.magnitude;
        
        bool isMovingForward = localVelocityZ > 0.1f; 

        if (speed == 0f) 
        {
            ApplyBrakeTorque(0f);
            ApplyMotorTorque(0f);
            return;
        }

        if (speed > 0f) 
        {
            ApplyBrakeTorque(0f);
            ApplyMotorTorque(currentSpeedMPS < maxSpeedMetersPerSec ? (speed * maxMotorTorque) : 0f);
            return;
        }

        if (speed < 0f) 
        {
            if (isMovingForward)
            {
                ApplyMotorTorque(0f);
                ApplyBrakeTorque(Mathf.Abs(speed) * maxBrakeTorque);
            }
            else 
            {
                ApplyBrakeTorque(0f);
                ApplyMotorTorque(speed * maxMotorTorque); 
            }
        }
    }

    public void StopMotor()
    {
        SetSpeed(0f);
    }

    public float GetCurrentSetSpeed()
    {
        return m_currentSetSpeed;
    }

    private void ApplyMotorTorque(float torque)
    {
        foreach (var wheel in m_wheels) wheel.motorTorque = torque;
    }

    private void ApplyBrakeTorque(float brakeForce)
    {
        foreach (var wheel in m_wheels) wheel.brakeTorque = brakeForce;
    }
}