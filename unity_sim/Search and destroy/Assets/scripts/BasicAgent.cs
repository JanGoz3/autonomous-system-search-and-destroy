using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

public class BasicAgent : Agent
{
    Rigidbody rBody;
    public float forceMultiplier = 10;

    private float previousDistance;

    void Start() {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin() {
        // resets agent position and movement
        if (transform.localPosition.y < 0) {
            rBody.angularVelocity = Vector3.zero;
            rBody.linearVelocity = Vector3.zero;
            transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        // randomizes location of the target
        float randomX = UnityEngine.Random.Range(-3f, 3f);
        float randomZ = UnityEngine.Random.Range(-3f, 3f);
        Target.localPosition = new Vector3(randomX, 0.5f, randomZ);

        previousDistance = Vector3.Distance(transform.localPosition, Target.localPosition);
    }

    public override void CollectObservations(VectorSensor sensor) {
        // Target and Agent positions
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(transform.localPosition);

        // Agent Velocity
        sensor.AddObservation(rBody.linearVelocity.x);
        sensor.AddObservation(rBody.linearVelocity.z);
    }

    public override void OnActionReceived(ActionBuffers actions) {
        // actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actions.ContinuousActions[0];
        controlSignal.z = actions.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        // Rewards
        float currentDistance = Vector3.Distance(transform.localPosition, Target.localPosition);    
        
        // 1. Reached Target (Big Reward)
        if (currentDistance < 1.42f) {
            SetReward(2.0f);
            EndEpisode();
        } 
        // 2. Fell off the platform
        else if (transform.localPosition.y < 0) {
            SetReward(-1.0f); 
            EndEpisode();
        }
        // 3. Still playing
        else {
            float distanceMoved = previousDistance - currentDistance;
            AddReward(distanceMoved); 
            // Time Penalty
            AddReward(-0.001f);
            previousDistance = currentDistance;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        var continuousActionsOut = actionsOut.ContinuousActions;
        var keyboard = Keyboard.current;
        
        if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed) 
            continuousActionsOut[0] = 1f;
        if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed) 
            continuousActionsOut[0] = -1f;

        // Replicate Input.GetAxis("Vertical")
        if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed) 
            continuousActionsOut[1] = 1f;
        if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed) 
            continuousActionsOut[1] = -1f;
    }
}
