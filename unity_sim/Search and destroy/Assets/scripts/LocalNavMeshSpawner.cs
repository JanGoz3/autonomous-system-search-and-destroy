using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class LocalNavMeshSpawner : MonoBehaviour
{
    [Header("Local Environment Binding")]
    public Transform buildingCenter;
    [Tooltip("How large is the building?")]
    public float buildingRadius = 30f;

    private List<Vector3> localVertices =  new List<Vector3>();
    private List<int> localIndices = new List<int>();

    void Awake() {
        FilterLocalNavMesh();
    }

    private void FilterLocalNavMesh() {
        NavMeshTriangulation globalNavMeshData = NavMesh.CalculateTriangulation();
        int totalTriangles = globalNavMeshData.indices.Length / 3;

        for (int i = 0; i < totalTriangles; i++) {
            int index1 = globalNavMeshData.indices[i * 3];
            int index2 = globalNavMeshData.indices[i * 3 + 1];
            int index3 = globalNavMeshData.indices[i * 3 + 2];

            Vector3 v1 = globalNavMeshData.vertices[index1];
            Vector3 v2 = globalNavMeshData.vertices[index2];
            Vector3 v3 = globalNavMeshData.vertices[index3];

            Vector3 triangleCenter = (v1 + v2 + v3) / 3f;

            if (Vector3.Distance(triangleCenter, buildingCenter.position) <= buildingRadius) {
                localIndices.Add(localVertices.Count);
                localVertices.Add(v1);
                localIndices.Add(localVertices.Count);
                localVertices.Add(v2);
                localIndices.Add(localVertices.Count);
                localVertices.Add(v3);
            }
        }

        if (localIndices.Count == 0) {
            Debug.LogError($"No NavMesh found for Environment {gameObject.name}! Check radius or NavMesh Bake.");
        }
    }

    public Vector3 GetRandomSafePoint() {
        if (localIndices.Count == 0) return buildingCenter.position;

        int traingleStartIndex = Random.Range(0, localIndices.Count / 3) * 3;

        Vector3 vertex1 = localVertices[localIndices[traingleStartIndex]];
        Vector3 vertex2 = localVertices[localIndices[traingleStartIndex + 1]];
        Vector3 vertex3 = localVertices[localIndices[traingleStartIndex + 2]];

        // Sampling a Parallelogram
        float r1 = Random.value;
        float r2 = Random.value;

        // Folding the Point Back
        if (r1 + r2 > 1f) {
            r1 = 1f - r1;
            r2 = 1f- r2;
        }

        // Converting to 3D World Coordinates
        return vertex1 + r1 * (vertex2 - vertex1) + r2 * (vertex3 - vertex1);
    }
}
