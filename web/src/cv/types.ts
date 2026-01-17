/**
 * Type definitions for the CV pipeline.
 */

export interface Point {
  x: number;
  y: number;
}

export type Corners = [Point, Point, Point, Point]; // TL, TR, BR, BL

export interface GridDetectionResult {
  corners: Corners;
  confidence: number;
}

export interface ProcessedCell {
  imageData: ImageData;
  row: number;
  col: number;
  isEmpty: boolean;
}
