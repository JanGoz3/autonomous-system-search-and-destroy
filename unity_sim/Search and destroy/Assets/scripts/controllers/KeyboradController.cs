using UnityEngine;
using UnityEngine.InputSystem;

public class KeyboardController : MonoBehaviour
{
    [Header("Chassis Reference")]
    public Chassis chassis;

    [Header("Sensitivity Settings")]
    public float steeringSensitivity = 3f;
    public float speedSensitivity = 2f;
    public float turretSensitivity = 4f;
    private float m_currentSteering = 0f;
    private float m_currentSpeed = 0f;
    private float m_currentPitch = 0f;
    private float m_currentYaw = 0f;

    void Update()
    {
        if (chassis == null) return;

        float targetSpeed = 0f;
        float targetSteering = 0f;
        float targetPitch = 0f;
        float targetYaw = 0f;

        if (Keyboard.current != null)
        {
            if (Keyboard.current.upArrowKey.isPressed) targetSpeed = 1f;
            if (Keyboard.current.downArrowKey.isPressed) targetSpeed = -1f;

            if (Keyboard.current.rightArrowKey.isPressed) targetSteering = 1f;
            if (Keyboard.current.leftArrowKey.isPressed) targetSteering = -1f;

            if (Keyboard.current.wKey.isPressed) targetPitch = 1f;
            if (Keyboard.current.sKey.isPressed) targetPitch = -1f;

            if (Keyboard.current.dKey.isPressed) targetYaw = 1f;
            if (Keyboard.current.aKey.isPressed) targetYaw = -1f;
        }

        m_currentSpeed = Mathf.MoveTowards(m_currentSpeed, targetSpeed, speedSensitivity * Time.deltaTime);
        m_currentSteering = Mathf.MoveTowards(m_currentSteering, targetSteering, steeringSensitivity * Time.deltaTime);
        m_currentPitch = Mathf.MoveTowards(m_currentPitch, targetPitch, turretSensitivity * Time.deltaTime);
        m_currentYaw = Mathf.MoveTowards(m_currentYaw, targetYaw, turretSensitivity * Time.deltaTime);

        chassis.SetSpeed(m_currentSpeed);
        chassis.SetSteering(m_currentSteering);
        chassis.SetCameraServos(m_currentPitch, m_currentYaw);
    }
}