using UnityEngine;
using System.Collections.Generic;

public class MockArTracker : MonoBehaviour
{
    [Header("Noise & Drift Profile")]
    public float positionJitterStdDev = 0.008f;
    public float yawDriftDegPer100m = 2.0f;
    public float scaleErrorFactor = 0.02f;
    [Header("Latency Simulation")]
    public float latencySeconds = 0.04f;

    private Vector3 m_SessionOrginPosition;
    private Quaternion m_SessionOrginRotation;

    private float m_TotalDistanceDriven = 0f;
    private Vector3 m_LastTruePosition;
    private float m_AccumulatedYawDrift = 0f;
    private float m_RandomScaleDrift = 1.0f;

    private struct PoseRecord
    {
        public float timestamp;
        public Vector3 position;
        public float yaw;
    }
    private readonly Queue<PoseRecord> m_PoseHistory = new Queue<PoseRecord>();

    public void ResetSession()
    {
        m_SessionOrginPosition = transform.position;
        m_SessionOrginRotation = transform.rotation;
        m_LastTruePosition = transform.position;

        m_TotalDistanceDriven = 0f;
        m_AccumulatedYawDrift = 0f;

        m_RandomScaleDrift = 1.0f + Random.Range(-scaleErrorFactor, scaleErrorFactor);
        m_PoseHistory.Clear();
    }

    void FixedUpdate()
    {
        Vector3 trueDeltaWorld = transform.position - m_SessionOrginPosition;
        Vector3 sessionPos = Quaternion.Inverse(m_SessionOrginRotation) * trueDeltaWorld;

        float trueYaw = (transform.rotation * Quaternion.Inverse(m_SessionOrginRotation)).eulerAngles.y;
        if (trueYaw > 180f) trueYaw -= 360f;

        float stepDist = Vector3.Distance(transform.position, m_LastTruePosition);
        m_TotalDistanceDriven += stepDist;
        m_LastTruePosition = transform.position;

        m_AccumulatedYawDrift += stepDist / 100f * yawDriftDegPer100m * (Random.value > 0.5f ? 1f : -1f);

        // Inject imperfections
        // Scale error + high frequency Gaussian-like jitter
        float jitterX = (Random.Range(-1f, 1f) + Random.Range(-1f, 1f)) * 0.5f * positionJitterStdDev;
        float jitterZ = (Random.Range(-1f, 1f) + Random.Range(-1f, 1f)) * 0.5f * positionJitterStdDev;

        Vector3 noisySessionPos = new Vector3(
            (sessionPos.x * m_RandomScaleDrift) + jitterX,
            0f,
            (sessionPos.z * m_RandomScaleDrift) + jitterZ
        );

        float noisyYaw = trueYaw + m_AccumulatedYawDrift + Random.Range(-0.2f, 0.2f);

        m_PoseHistory.Enqueue(new PoseRecord
        {
            timestamp = Time.time,
            position = noisySessionPos,
            yaw = noisyYaw
        });
    }

    /// <summary>
    /// Calculates the remaining distance to the waypoint from the car's local perspective 
    /// (X = right/left, Z = forward/backward), exactly how a physical phone's AR tracker 
    /// would calculate it in the real world.
    /// </summary>
    public Vector2 GetRelativeTargetVector(Vector3 targetWorldPosition)
    {
        float targetTime = Time.time - latencySeconds;
        PoseRecord currentPose = m_PoseHistory.Count > 0 ? m_PoseHistory.Peek() : default;

        while (m_PoseHistory.Count > 1 && m_PoseHistory.Peek().timestamp < targetTime)
        {
            currentPose = m_PoseHistory.Dequeue();
        } 

        // Convert target to Session Space
        Vector3 targetDeltaWorld = targetWorldPosition - m_SessionOrginPosition;
        Vector3 targetSessionPos = Quaternion.Inverse(m_SessionOrginRotation) * targetDeltaWorld;

        // Delta in session space
        float dx = targetSessionPos.x - currentPose.position.x;
        float dz = targetSessionPos.z - currentPose.position.z;

        // rotate by the car's noisy heading (-yaw) to bring into car local space
        float angleRad = -currentPose.yaw * Mathf.Deg2Rad;
        float cos = Mathf.Cos(angleRad);
        float sin = Mathf.Sin(angleRad);

        float localX = dx * cos - dz * sin;
        float localZ = dx * sin + dz * cos;

        return new Vector2(localX, localZ);
    }
}
