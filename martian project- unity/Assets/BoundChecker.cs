using UnityEngine;

public class BoundsChecker : MonoBehaviour
{
    public Renderer targetRenderer;

    void Start()
    {
        Bounds b = targetRenderer.bounds;
        Debug.Log($"World bounds center: {b.center}");
        Debug.Log($"World bounds size:   {b.size}");
        Debug.Log($"World bounds min:    {b.min}");
        Debug.Log($"World bounds max:    {b.max}");
    }
}