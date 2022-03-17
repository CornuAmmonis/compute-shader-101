struct Params {
    width: u32;
    height: u32;
    frame: u32;
};

struct StorageBuffer {
    data: array<atomic<u32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var col: texture_storage_2d<rgba16float,write>;
[[group(0), binding(2)]] var<storage,read_write> buf: StorageBuffer;
[[group(0), binding(3)]] var tex: texture_2d_array<f32>;
[[group(0), binding(4)]] var texs: texture_storage_2d_array<rgba16float,write>;
[[group(0), binding(5)]] var nearest: sampler;
[[group(0), binding(6)]] var bilinear: sampler;

// https://www.jcgt.org/published/0009/03/02/
// https://www.pcg-random.org/
fn pcg(seed: ptr<function, u32>) -> f32 {
	*seed = *seed * 747796405u + 2891336453u;
	let word = ((*seed >> ((*seed >> 28u) + 4u)) ^ *seed) * 277803737u;
	return f32((word >> 22u) ^ word) / f32(0xffffffffu);
}

fn smoothstep(edge0: vec3<f32>, edge1: vec3<f32>, x: vec3<f32>) -> vec3<f32> {
    let t = clamp((x - edge0) / (edge1 - edge0), vec3<f32>(0.0), vec3<f32>(1.0));
    return t * t * (3.0 - 2.0 * t);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_hist([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    var seed = global_id.x + global_id.y * params.width + params.frame * params.width * params.height;
    let uv = vec2<f32>(f32(global_id.x) + pcg(&seed), f32(global_id.y) + pcg(&seed)) / resolution.xy;
    let aspect = resolution.xy / resolution.y;
    let c = (uv * 2. - 1.) * aspect - vec2<f32>(.5, 0.);
    var z = vec2<f32>(0.);
    var n = 0;
    for (n = 0; n < 5000; n = n + 1) {
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
        if (dot(z,z) > 9.) { break; }
    }
    z = vec2<f32>(0.);
    for (var i = 0; i < 5000; i = i + 1) {
        z = vec2<f32>(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z,z) > 9.) { break; }
        let p = (z + vec2<f32>(.5, 0.)) / aspect * .5 + .5;
        let id1 = u32(resolution.x * p.x) + u32(resolution.y * p.y) * params.width;
        let id2 = u32(resolution.x * p.x) + u32(resolution.y * (1. - p.y)) * params.width;
        if (n < 50) {
            atomicAdd(&buf.data[id1*4u+0u], 1u);
            atomicAdd(&buf.data[id2*4u+0u], 1u);
        } else if (n < 500) {
            atomicAdd(&buf.data[id1*4u+1u], 1u);
            atomicAdd(&buf.data[id2*4u+1u], 1u);
        } else if (n < 5000) {
            atomicAdd(&buf.data[id1*4u+2u], 1u);
            atomicAdd(&buf.data[id2*4u+2u], 1u);
        }
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_image([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let id = global_id.x + global_id.y * params.width;
    let x = f32(atomicLoad(&buf.data[id*4u+2u]));
    let y = f32(atomicLoad(&buf.data[id*4u+1u]));
    let z = f32(atomicLoad(&buf.data[id*4u+0u]));
    var r = vec3<f32>(x + y + z, y + z, z) / f32(65u * params.frame);
    r = smoothstep(vec3<f32>(0.), vec3<f32>(1.), 2.5 * pow(r, vec3<f32>(1.5, 1.4, 1.3)));
    textureStore(col, vec2<i32>(global_id.xy), vec4<f32>(r, 1.));
}