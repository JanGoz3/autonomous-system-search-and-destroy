using UnityEngine;

public class Steering : MonoBehaviour
{
    [Header("Steering Setup (Traxxas 2075 Servo)")]
    public float steeringServoSpeedDegPerSec = 350f;

    private AnimationCurve m_physicalSteeringCurve;

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

        if (m_physicalSteeringCurve == null || m_physicalSteeringCurve.length == 0)
        {
            m_physicalSteeringCurve = new AnimationCurve(
                new Keyframe(-1f, -23.5f),
                new Keyframe(-0.5f, -14f),
                new Keyframe(0f, 0f),
                new Keyframe(0.5f, 14f),
                new Keyframe(1f, 25f)
            );
        }

        m_currentSteerAngle = m_physicalSteeringCurve.Evaluate(0f);
        m_targetSteerAngle = m_currentSteerAngle;
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

        m_targetSteerAngle = m_physicalSteeringCurve.Evaluate(swing);
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