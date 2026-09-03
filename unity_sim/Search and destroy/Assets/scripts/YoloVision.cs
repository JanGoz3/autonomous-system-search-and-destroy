using UnityEngine;
using Unity.InferenceEngine;
using System.Collections.Generic;

public class YoloVision : MonoBehaviour
{
    [Header("Inference Config")]
    public ModelAsset yoloModelAsset;
    public RenderTexture cameraRenderTexture;
    public Camera yoloCamera;
    private Worker m_Worker;
    private Tensor<float> m_InputTensor;
    private const int InferenceInterval = 10; // Run every nth physics tick
    private int m_StepCounter = 0;
    private const int MaxTrackedObjects = 3;
    private const int FeaturesPerObject = 9; // 5 spatial + 4 one-hot classes.
    private readonly float[] m_LatestYoloState = new float[MaxTrackedObjects * FeaturesPerObject];
    private const int CLASS_TARGET = 0;
    private const int CLASS_PERSON = 1;
    private const int CLASS_CHAIR = 2;
    private const int CLASS_DOOR = 3;

    private struct Detection {
        public float x, y, w, h, conf, classId;
        public float area => w * h;
    }
    private readonly List<Detection> m_Detections = new List<Detection>(300);

    void Start() 
    {
        if (yoloCamera.enabled) 
        {
            Debug.LogWarning("Camera component is active! this will result in frames getting rendered twice. Make sure the camera component is disabled in inspector");
        }
        var model = ModelLoader.Load(yoloModelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);  
        m_InputTensor = new Tensor<float>(new TensorShape(1, 3, 320, 320));
    }

    public float[] GetYoloState() 
    {
        // run inference every nth frame
        if (m_StepCounter % InferenceInterval == 0)
        {
            RunInference();
        }
        m_StepCounter++;
        return m_LatestYoloState;
    }

    private void RunInference() 
    {
        if (cameraRenderTexture == null || yoloCamera == null) return;    

        yoloCamera.Render();
        TextureConverter.ToTensor(cameraRenderTexture, m_InputTensor, new TextureTransform());
        m_Worker.Schedule(m_InputTensor);

        var outputTensor = m_Worker.PeekOutput() as Tensor<float>;     
        float[] rawOutput = outputTensor.DownloadToArray();

        System.Array.Clear(m_LatestYoloState, 0, m_LatestYoloState.Length);

        m_Detections.Clear();

        int numBoxes = 300;
        int features = 6;
        float confThreshold = 0.5f;

        for (int i = 0; i < numBoxes; i++) 
        {
            float conf = rawOutput[i * features + 4];

            if (conf > confThreshold) 
            {
                m_Detections.Add(new Detection 
                {
                    // center normalized [-1.0, 1.0], where 0 is dead center
                    // this supposedly makes the network converge faster
                    x = (rawOutput[i * features + 0] - 160f) / 160f, 
                    y = (rawOutput[i * features + 1] - 160) / 160f, 
                    
                    // size-normalized: [0.0, 1.0]
                    w = rawOutput[i * features + 2] / 320f, 
                    h = rawOutput[i * features + 3] / 320f,
                    conf = conf,
                    classId = rawOutput[i * features + 5] 
                });
            }
        }

        // sort by largest area first (closest/most prominent hazards)
        m_Detections.Sort((a,b) => b.area.CompareTo(a.area));

        int count = Mathf.Min(m_Detections.Count, MaxTrackedObjects);
        for (int i = 0; i < count; i++) 
        {
            int offset = i * FeaturesPerObject;
            var d = m_Detections[i];

            // Spatial & Confidence
            m_LatestYoloState[offset + 0] = d.x;    
            m_LatestYoloState[offset + 1] = d.y;    
            m_LatestYoloState[offset + 2] = d.w;    
            m_LatestYoloState[offset + 3] = d.h;    
            m_LatestYoloState[offset + 4] = d.conf;

            // One-Hot Class Flags
            int id = Mathf.RoundToInt(d.classId);
            m_LatestYoloState[offset + 5] = (id == CLASS_TARGET) ? 1f : 0f;
            m_LatestYoloState[offset + 6] = (id == CLASS_PERSON) ? 1f : 0f;
            m_LatestYoloState[offset + 7] = (id == CLASS_CHAIR)  ? 1f : 0f;
            m_LatestYoloState[offset + 8] = (id == CLASS_DOOR)   ? 1f : 0f;
        }
    }       

    void OnDestroy()
    {
        m_Worker?.Dispose();
        m_InputTensor?.Dispose();
    } 
}
