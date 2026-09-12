using UnityEngine;
using System.Collections.Generic;

public class Shooter : MonoBehaviour
{

    [Header("Shooter config")]
    public YoloVision yolo;
    public CarAgent carAgent;
    [Header("Engagement Settings")]
    public float engageWidthThreshold = 40f;
    public float trackingSensitivity = 2.0f;
    
    [Header("Debug")]
    public bool showDebugVision = true;
    public RenderTexture cameraRenderTexture;

    void FixedUpdate()
    {
        if (yolo == null || carAgent == null) return;

        bool targetFound = false;
        YoloVision.DetectionRaw bestTarget = default;
        float largestArea = 0f;

        // find the largest target in view
        foreach(var d in yolo.detectionRaw)
        {
            if (d.classId == YoloVision.CLASS_TARGET)
            {
                float width = d.xMax - d.xMin;
                float height = d.yMax - d.yMin;
                float area = width * height;

                if (width >= engageWidthThreshold && area > largestArea)
                {
                    targetFound = true;
                    bestTarget = d;
                    largestArea = area;
                }
            }
        }
        if (targetFound)
        {
            if (!carAgent.isEngagingTarget)
            {
                (carAgent.autoAimPitch, carAgent.autoAimYaw) = carAgent.chassis.servosCamera.GetActualNormalizedSwing();
            }

            carAgent.isEngagingTarget = true;
            float targetCenterX = bestTarget.xMin +((bestTarget.xMax - bestTarget.xMin) / 2f);
            float targetCenterY = bestTarget.yMin + ((bestTarget.yMax - bestTarget.yMin) / 2f);    

            float errorX = (targetCenterX - 160f) / 160f;
            float errorY = (targetCenterY - 160f) / 160f;

            carAgent.autoAimYaw += errorX * trackingSensitivity * Time.fixedDeltaTime;
            carAgent.autoAimPitch -= errorY * trackingSensitivity * Time.fixedDeltaTime;
            carAgent.autoAimYaw = Mathf.Clamp(carAgent.autoAimYaw, -1f, 1f);
            carAgent.autoAimPitch = Mathf.Clamp(carAgent.autoAimPitch, -1f, 1f);
        }
        else
        {
           carAgent.isEngagingTarget = false;    
        }
    }

    void OnGUI()
    {
        if (!showDebugVision || cameraRenderTexture == null || yolo.detectionRaw == null) return;

        Rect displayRect = new Rect(0, 0, Screen.width, Screen.height);

        GUI.DrawTexture(displayRect, cameraRenderTexture, ScaleMode.StretchToFill);

        float scaleX = Screen.width / 320f;
        float scaleY = Screen.height / 320f;

        foreach (var d in yolo.detectionRaw)
        {
            if (d.classId == YoloVision.CLASS_TARGET)
            {
                //Debug.Break();
            }
            // Direct calculation from the raw pixels stored in the debug struct
            float boxWidth = (d.xMax - d.xMin) * scaleX;
            float boxHeight = (d.yMax - d.yMin) * scaleY;

            float drawX = d.xMin;
            float drawY = d.yMin;

            Rect boxRect = new Rect(
                displayRect.x + (drawX * scaleX) , 
                displayRect.y + (drawY * scaleY) , 
                boxWidth, 
                boxHeight
            );

            Color boxColor = Color.white;
            string label = "Unknown";
            int id = Mathf.RoundToInt(d.classId);
            
            if (id == YoloVision.CLASS_TARGET) { boxColor = Color.red; label = "Target"; }
            else if (id == YoloVision.CLASS_PERSON) { boxColor = Color.blue; label = "Person"; }
            else if (id == YoloVision.CLASS_CHAIR) { boxColor = Color.yellow; label = "Chair"; }
            else if (id == YoloVision.CLASS_DOOR) { boxColor = Color.magenta; label = "Door"; }

            DrawBoxBorder(boxRect, boxColor, 2f);
            GUI.contentColor = boxColor;
            GUI.Label(new Rect(boxRect.x, boxRect.y - 20, 150, 20), $"{label} ({d.conf:F2})");
        }
    }

    // Helper method to draw hollow rectangles in OnGUI
    private void DrawBoxBorder(Rect rect, Color color, float thickness)
    {
        Texture2D tex = Texture2D.whiteTexture;
        Color oldColor = GUI.color;
        GUI.color = color;
        
        // Top line
        GUI.DrawTexture(new Rect(rect.x, rect.y, rect.width, thickness), tex);
        // Bottom line
        GUI.DrawTexture(new Rect(rect.x, rect.y + rect.height - thickness, rect.width, thickness), tex);
        // Left line
        GUI.DrawTexture(new Rect(rect.x, rect.y, thickness, rect.height), tex);
        // Right line
        GUI.DrawTexture(new Rect(rect.x + rect.width - thickness, rect.y, thickness, rect.height), tex);
        
        GUI.color = oldColor;
    }
}
