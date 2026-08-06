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
}