/**
 * Camera capture module for browser-based sudoku scanning.
 * Handles webcam/phone camera access and frame capture.
 */

export class Camera {
  private video: HTMLVideoElement;
  private stream: MediaStream | null = null;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(videoElement: HTMLVideoElement) {
    this.video = videoElement;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
  }

  /**
   * Start the camera stream.
   * Prefers back camera on mobile devices.
   */
  async start(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Prefer back camera
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
        audio: false,
      });

      this.video.srcObject = this.stream;
      await this.video.play();

      // Wait for video to be ready
      await new Promise<void>((resolve) => {
        if (this.video.readyState >= 2) {
          resolve();
        } else {
          this.video.onloadeddata = () => resolve();
        }
      });

      // Set canvas size to match video
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    } catch (error) {
      if (error instanceof DOMException) {
        if (error.name === 'NotAllowedError') {
          throw new Error('Camera permission denied. Please allow camera access.');
        }
        if (error.name === 'NotFoundError') {
          throw new Error('No camera found on this device.');
        }
      }
      throw error;
    }
  }

  /**
   * Stop the camera stream.
   */
  stop(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.video.srcObject = null;
  }

  /**
   * Capture current video frame as ImageData.
   */
  captureFrame(): ImageData {
    this.ctx.drawImage(this.video, 0, 0);
    return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Capture current video frame as HTMLCanvasElement.
   * Useful for OpenCV.js which can read directly from canvas.
   */
  captureCanvas(): HTMLCanvasElement {
    this.ctx.drawImage(this.video, 0, 0);
    return this.canvas;
  }

  /**
   * Get video dimensions.
   */
  getDimensions(): { width: number; height: number } {
    return {
      width: this.video.videoWidth,
      height: this.video.videoHeight,
    };
  }

  /**
   * Check if camera is active.
   */
  isActive(): boolean {
    return this.stream !== null && this.stream.active;
  }
}
