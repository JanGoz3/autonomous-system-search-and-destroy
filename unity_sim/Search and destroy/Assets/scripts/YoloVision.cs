using UnityEngine;
using Unity.InferenceEngine;
 
public class YoloVision : MonoBehaviour
{
    [Header("Inference Config")]
    public ModelAsset yoloModelAsset;
    public RenderTexture cameraRenderTexture;
    public Camera yoloCamera;

    private Worker m_Worker;
    private float[] m_LatestYoloState;
    private Tensor<float> m_InputTensor;
    private const int InferenceInterval = 10; // Run every nth physics tick
    private int m_StepCounter = 0;

    void Start() 
    {
        var model = ModelLoader.Load(yoloModelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);
        m_LatestYoloState = new float[6];    
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

        int numBoxes = 300;
        int features = 6;
        float maxConfidence = 0.5f;
        int bestBoxIndex = -1;

        for (int i = 0; i < numBoxes; i++) {
            float confidence = rawOutput[i * features + 4];

            if (confidence > maxConfidence) {
                maxConfidence = confidence;
                bestBoxIndex = i;
            }
        }

        // 5. Extract the data if we found something
        if (bestBoxIndex != -1)
        {
            float x = rawOutput[bestBoxIndex * features + 0];
            float y = rawOutput[bestBoxIndex * features + 1];
            float w = rawOutput[bestBoxIndex * features + 2];
            float h = rawOutput[bestBoxIndex * features + 3];
            float conf = rawOutput[bestBoxIndex * features + 4];
            float classId = rawOutput[bestBoxIndex * features + 5];

            // Divide the coordinates by 320 to normalize them from 0.0 to 1.0
            m_LatestYoloState[0] = x / 320f; 
            m_LatestYoloState[1] = y / 320f; 
            m_LatestYoloState[2] = w / 320f;  
            m_LatestYoloState[3] = h / 320f;  
            m_LatestYoloState[4] = conf;
            m_LatestYoloState[5] = classId; 
        }
        
    }       

    void OnDestroy()
    {
        m_Worker?.Dispose();
        m_InputTensor?.Dispose();
    } 
}
