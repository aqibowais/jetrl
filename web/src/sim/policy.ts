/**
 * Deterministic forward pass of an SB3 PPO MlpPolicy actor:
 * Linear -> Tanh -> Linear -> Tanh -> Linear (action mean), clipped to [-1, 1].
 * Weights come from export_web.py as base64 float32, row-major [out][in].
 */

export interface LayerJson {
  in: number;
  out: number;
  w: string;
  b: string;
}

export interface PolicyJson {
  id: string;
  layers: LayerJson[];
  /** Trained with the gate-index observation zeroed (generalist policies). */
  mask?: boolean;
}

interface Layer {
  inDim: number;
  outDim: number;
  w: Float32Array;
  b: Float32Array;
}

function decode(b64: string): Float32Array {
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export class Policy {
  readonly id: string;
  readonly mask: boolean;
  private readonly layers: Layer[];
  /** Post-activation values of each hidden layer from the last call, for the network view. */
  readonly hidden: Float32Array[];

  constructor(json: PolicyJson) {
    this.id = json.id;
    this.mask = !!json.mask;
    this.layers = json.layers.map((layer) => {
      const w = decode(layer.w);
      const b = decode(layer.b);
      if (w.length !== layer.in * layer.out || b.length !== layer.out) {
        throw new Error(`policy ${json.id}: bad layer shape`);
      }
      return { inDim: layer.in, outDim: layer.out, w, b };
    });
    this.hidden = this.layers.slice(0, -1).map((layer) => new Float32Array(layer.outDim));
  }

  get inputSize(): number {
    return this.layers[0].inDim;
  }

  act(obs: ArrayLike<number>, out: number[] = [0, 0, 0, 0]): number[] {
    let x: ArrayLike<number> = obs;
    const last = this.layers.length - 1;
    for (let li = 0; li <= last; li += 1) {
      const { inDim, outDim, w, b } = this.layers[li];
      const y = li === last ? new Float32Array(outDim) : this.hidden[li];
      for (let o = 0; o < outDim; o += 1) {
        let acc = b[o];
        const row = o * inDim;
        for (let i = 0; i < inDim; i += 1) acc += w[row + i] * x[i];
        y[o] = li === last ? acc : Math.tanh(acc);
      }
      x = y;
    }
    for (let i = 0; i < out.length; i += 1) out[i] = Math.min(1, Math.max(-1, x[i]));
    return out;
  }
}
