using UnityEngine;

public class Steering : MonoBehaviour
{
    [Header("Steering Setup (Traxxas 2075 Servo)")]
    public float steeringServoSpeedDegPerSec = 350f;
    public float physicalLeftSteerAngle = -35f;  
    public float physicalCenterSteerAngle = 3f;  
    public float physicalRightSteerAngle = 27f;  

    private WheelCollider m_frontLeft;
    private WheelCollider m_frontRight;
    private bool m_isInitialized = false;

    private float m_currentSteerAngle = 0f;
    private float m_targetSteerAngle = 0f;
    private float m_currentSetSwing = 0.0f;

    public void Initialize(WheelCollider frontLeft, WheelCollider frontRight)
    {
        m_frontLeft = frontLeft;
        m_frontRight = frontRight;
        m_currentSteerAngle = physicalCenterSteerAngle;
        m_targetSteerAngle = physicalCenterSteerAngle;
        m_isInitialized = true;
    }

    void Update()
    {
        if (!m_isInitialized) return;

        m_currentSteerAngle = Mathf.MoveTowards(m_currentSteerAngle, m_targetSteerAngle, steeringServoSpeedDegPerSec * Time.deltaTime);
        
        m_frontLeft.steerAngle = m_currentSteerAngle;
        m_frontRight.steerAngle = m_currentSteerAngle;
    }

    public void SetSteering(float swing)
    {
        if (!m_isInitialized) return;

        swing = Mathf.Clamp(swing, -1f, 1f);
        m_currentSetSwing = swing;

        if (swing < 0f)
            m_targetSteerAngle = Mathf.Lerp(physicalLeftSteerAngle, physicalCenterSteerAngle, swing + 1f);
        else
            m_targetSteerAngle = Mathf.Lerp(physicalCenterSteerAngle, physicalRightSteerAngle, swing);
    }

    public void SetNeutralSwing()
    {
        SetSteering(0f);
    }

    public float GetCurrentSetSwing()
    {
        return m_currentSetSwing;
    }
}