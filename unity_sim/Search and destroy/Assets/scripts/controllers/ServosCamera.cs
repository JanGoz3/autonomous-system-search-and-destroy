using UnityEngine;

public class ServosCamera : MonoBehaviour
{
    [Header("Servos Transforms")]
    public Transform yawTransform;
    public Transform pitchTransform;

    [Header("Hardware Speeds")]
    public float servoSpeedDegPerSec = 90f;

    [Header("Yaw Calibration")]
    public float physicalLeftYawAngle = -87.5f;
    public float physicalCenterYawAngle = -3.5f;
    public float physicalRightYawAngle = 76.5f;

    [Header("Pitch Calibration")]
    public float physicalDownPitchAngle = 22f;
    public float physicalCenterPitchAngle = -6f;
    public float physicalUpPitchAngle = -40f;

    [Header("Test Sliders")]
    public bool enableManualTesting = false;

    [ContextMenu("Reset to Neutral Swing")]
    public void SetNeutralSwingButton()
    {
        testSwingPitch = 0f;
        testSwingYaw = 0f;
        SetNeutralSwing();
    }

    [Range(-1f, 1f)]
    public float testSwingPitch = 0f;

    [Range(-1f, 1f)]
    public float testSwingYaw = 0f;

    private float m_currentSetSwingPitch = 0.0f;
    private float m_currentSetSwingYaw = 0.0f;
    private bool m_isInitialized = false;

    private float m_targetYawAngle = 0f;
    private float m_currentYawAngle = 0f;
    
    private float m_targetPitchAngle = 0f;
    private float m_currentPitchAngle = 0f;

    public void Initialize()
    {
        m_isInitialized = true;
        SetNeutralSwing();
        
        m_currentYawAngle = physicalCenterYawAngle;
        m_targetYawAngle = physicalCenterYawAngle;
        
        m_currentPitchAngle = physicalCenterPitchAngle;
        m_targetPitchAngle = physicalCenterPitchAngle;
        
        ApplyRotations();
    }

    void FixedUpdate()
    {
        if (enableManualTesting)
        {
            SetPitchYaw(testSwingPitch, testSwingYaw); 
        }

        m_currentYawAngle = Mathf.MoveTowards(m_currentYawAngle, m_targetYawAngle, servoSpeedDegPerSec * Time.fixedDeltaTime);
        m_currentPitchAngle = Mathf.MoveTowards(m_currentPitchAngle, m_targetPitchAngle, servoSpeedDegPerSec * Time.fixedDeltaTime);

        ApplyRotations();
    }

    private void ApplyRotations()
    {
        if (yawTransform != null)
            yawTransform.localEulerAngles = new Vector3(0f, m_currentYawAngle, 0f);

        if (pitchTransform != null)
            pitchTransform.localEulerAngles = new Vector3(m_currentPitchAngle, 0f, 0f);
    }

    public void SetPitchYaw(float swingPitch, float swingYaw)
    {
        SetPitch(swingPitch);
        SetYaw(swingYaw);
    }

    public void SetPitch(float swingPitch)
    {
        if (!m_isInitialized) return;

        swingPitch = Mathf.Clamp(swingPitch, -1f, 1f);
        m_currentSetSwingPitch = swingPitch;

        if (swingPitch < 0f)
            m_targetPitchAngle = Mathf.Lerp(physicalDownPitchAngle, physicalCenterPitchAngle, swingPitch + 1f);
        else
            m_targetPitchAngle = Mathf.Lerp(physicalCenterPitchAngle, physicalUpPitchAngle, swingPitch);
    }

    public void SetYaw(float swingYaw)
    {
        if (!m_isInitialized) return;

        swingYaw = Mathf.Clamp(swingYaw, -1f, 1f);
        m_currentSetSwingYaw = swingYaw;

        if (swingYaw < 0f)
            m_targetYawAngle = Mathf.Lerp(physicalLeftYawAngle, physicalCenterYawAngle, swingYaw + 1f);
        else
            m_targetYawAngle = Mathf.Lerp(physicalCenterYawAngle, physicalRightYawAngle, swingYaw);
    }

    public void SetNeutralSwing()
    {
        if (!m_isInitialized) return;
        SetPitchYaw(0f, 0f);
    }

    public (float pitch, float yaw) GetCurrentPitchYaw()
    {
        return (m_currentSetSwingPitch, m_currentSetSwingYaw);
    }

    public (float pitch, float yaw) GetActualPitchYawDegrees()
    {
        return (m_currentPitchAngle, m_currentYawAngle);
    }

    public (float pitch, float yaw) GetActualNormalizedSwing()
    {
        float normYaw = (m_currentYawAngle < physicalCenterYawAngle)
            ? Mathf.InverseLerp(physicalLeftYawAngle, physicalCenterYawAngle, m_currentYawAngle) - 1
            : Mathf.InverseLerp(physicalCenterYawAngle, physicalRightYawAngle, m_currentYawAngle);

        float normPitch = (m_currentPitchAngle > physicalCenterPitchAngle)
            ? Mathf.InverseLerp(physicalDownPitchAngle, physicalCenterPitchAngle, m_currentPitchAngle) - 1f
            : Mathf.InverseLerp(physicalCenterPitchAngle, physicalUpPitchAngle, m_currentPitchAngle);

        return (Mathf.Clamp(normPitch, -1f, 1f), Mathf.Clamp(normYaw, -1f, 1f));
    }
}
