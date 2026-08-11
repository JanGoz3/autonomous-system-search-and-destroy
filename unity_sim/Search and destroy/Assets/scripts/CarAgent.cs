using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

public class CarAgent : Agent
{
    [Header("Hardware Link")]
    public Chassis chassis;

    [Header("Training Environment")]
    public Transform startingPoint;

    [Header("Target Object")]
    public Transform Target;

    [Header("Heuristic Smoothing (Keyboard Only)")]
    public float steeringSensitivity = 3f;
    public float speedSensitivity = 0.25f;
    public float turretSensitivity = 4f;

    private float m_currentSteering = 0f;
    private float m_currentSpeed = 0f;
    private float m_currentPitch = 0f;
    private float m_currentYaw = 0f;
    private float previousDistance = 0f;

    public override void OnEpisodeBegin()
    {
        if (chassis != null)
        {
            chassis.SetNeutral();
            
            if (chassis.carRigidbody != null)
            {
                chassis.carRigidbody.linearVelocity = Vector3.zero;
                chassis.carRigidbody.angularVelocity = Vector3.zero;
            }
        }

        if (startingPoint != null)
        {
            transform.position = startingPoint.position;
            transform.rotation = startingPoint.rotation;
        }
        else
        {
            transform.localPosition = new Vector3(0, 0.5f, 0);
            transform.localRotation = Quaternion.identity;
        }

        // randomizes location of the target
        float randomX = UnityEngine.Random.Range(-9f, 9f);
        float randomZ = UnityEngine.Random.Range(-9f, 9f);
        Target.localPosition = new Vector3(randomX, 0.5f, randomZ);

        previousDistance = Vector3.Distance(transform.localPosition, Target.localPosition);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (chassis == null) return;

        float[] telemetryData = chassis.GetTelemetryState();

        foreach (float data in telemetryData)
        {
            sensor.AddObservation(data);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float aiMotor = actions.ContinuousActions[0];
        float aiSteering = actions.ContinuousActions[1];
        float aiCamPitch = actions.ContinuousActions[2];
        float aiCamYaw = actions.ContinuousActions[3];

        if (chassis != null)
        {
            chassis.SetSpeed(aiMotor);
            chassis.SetSteering(aiSteering);
            chassis.SetCameraServos(aiCamPitch, aiCamYaw);
        }

        // TODO: Rewards system
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        var keyboard = Keyboard.current;
        
        if (keyboard == null) return;

        float targetSpeed = 0f;
        float targetSteering = 0f;
        float targetPitch = 0f;
        float targetYaw = 0f;

        if (keyboard.upArrowKey.isPressed) targetSpeed = 1f;
        if (keyboard.downArrowKey.isPressed) targetSpeed = -1f;

        if (keyboard.rightArrowKey.isPressed) targetSteering = 1f;
        if (keyboard.leftArrowKey.isPressed) targetSteering = -1f;

        if (keyboard.wKey.isPressed) targetPitch = 1f;
        if (keyboard.sKey.isPressed) targetPitch = -1f;

        if (keyboard.dKey.isPressed) targetYaw = 1f;
        if (keyboard.aKey.isPressed) targetYaw = -1f;

        m_currentSpeed = Mathf.MoveTowards(m_currentSpeed, targetSpeed, speedSensitivity * Time.deltaTime);
        m_currentSteering = Mathf.MoveTowards(m_currentSteering, targetSteering, steeringSensitivity * Time.deltaTime);
        m_currentPitch = Mathf.MoveTowards(m_currentPitch, targetPitch, turretSensitivity * Time.deltaTime);
        m_currentYaw = Mathf.MoveTowards(m_currentYaw, targetYaw, turretSensitivity * Time.deltaTime);

        continuousActionsOut[0] = m_currentSpeed;
        continuousActionsOut[1] = m_currentSteering;
        continuousActionsOut[2] = m_currentPitch;
        continuousActionsOut[3] = m_currentYaw;
    }
}