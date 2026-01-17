/**
 * Digit classification using ONNX Runtime Web.
 * Runs the trained CNN model for digit recognition.
 */

import * as ort from 'onnxruntime-web';
import { preprocessCell, preprocessCellSimple } from './preprocessor';

export class DigitClassifier {
  private session: ort.InferenceSession | null = null;
  private inputName: string = 'input';
  private useSimplePreprocessing: boolean = false;

  /**
   * Load the ONNX model.
   */
  async load(modelPath: string = '/models/digit_cnn_v2.onnx'): Promise<void> {
    try {
      // Configure ONNX Runtime to use WebAssembly backend
      ort.env.wasm.numThreads = 1;

      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
      });

      // Get input name from model
      const inputNames = this.session.inputNames;
      if (inputNames.length > 0) {
        this.inputName = inputNames[0];
      }

      console.log('DigitClassifier loaded:', {
        inputNames: this.session.inputNames,
        outputNames: this.session.outputNames,
      });
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  /**
   * Enable simple preprocessing (fallback when OpenCV CLAHE unavailable).
   */
  setSimplePreprocessing(enabled: boolean): void {
    this.useSimplePreprocessing = enabled;
  }

  /**
   * Classify a single cell image.
   * @param imageData - 28x28 cell image
   * @returns Predicted digit (0-9) and confidence
   */
  async classifyCell(imageData: ImageData): Promise<{ digit: number; confidence: number }> {
    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // Preprocess the cell
    let input: Float32Array;
    try {
      input = this.useSimplePreprocessing
        ? preprocessCellSimple(imageData)
        : preprocessCell(imageData);
    } catch {
      // Fallback to simple preprocessing if OpenCV fails
      input = preprocessCellSimple(imageData);
    }

    // Create tensor with shape [1, 1, 28, 28]
    const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);

    // Run inference
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.inputName] = tensor;
    const results = await this.session.run(feeds);

    // Get output (assuming first output is logits/probabilities)
    const outputName = this.session.outputNames[0];
    const output = results[outputName];
    const data = output.data as Float32Array;

    // Find argmax and compute softmax for confidence
    let maxIdx = 0;
    let maxVal = data[0];
    for (let i = 1; i < data.length; i++) {
      if (data[i] > maxVal) {
        maxVal = data[i];
        maxIdx = i;
      }
    }

    // Compute softmax confidence
    const expValues = Array.from(data).map((x) => Math.exp(x - maxVal));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    const confidence = expValues[maxIdx] / sumExp;

    return { digit: maxIdx, confidence };
  }

  /**
   * Classify multiple cells in batch.
   * @param cells - Array of 28x28 cell ImageData
   * @returns Array of predictions with digit and confidence
   */
  async classifyCells(
    cells: { imageData: ImageData; isEmpty: boolean }[]
  ): Promise<{ digit: number; confidence: number }[]> {
    const results: { digit: number; confidence: number }[] = [];

    for (const cell of cells) {
      if (cell.isEmpty) {
        // Empty cell = 0 (no digit)
        results.push({ digit: 0, confidence: 1.0 });
      } else {
        const prediction = await this.classifyCell(cell.imageData);
        results.push(prediction);
      }
    }

    return results;
  }

  /**
   * Check if model is loaded.
   */
  isLoaded(): boolean {
    return this.session !== null;
  }
}
