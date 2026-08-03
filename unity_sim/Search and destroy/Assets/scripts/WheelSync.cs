using UnityEngine;

public class WheelSync : MonoBehaviour
{
    public WheelCollider wheelCollider;
    public Transform visualMeshContainer;

    void Update()
    {
        if (visualMeshContainer == null) return;

        Vector3 position;
        Quaternion rotation;
        wheelCollider.GetWorldPose(out position, out rotation);

        visualMeshContainer.position = position;
        visualMeshContainer.rotation = rotation;
    }
}