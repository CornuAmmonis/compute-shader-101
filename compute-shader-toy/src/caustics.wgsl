type int = i32;
type uint = u32;
type float = f32;

type int2 = vec2<i32>;
type int3 = vec3<i32>;
type int4 = vec4<i32>;
type uint2 = vec2<u32>;
type uint3 = vec3<u32>;
type uint4 = vec4<u32>;
type float2 = vec2<f32>;
type float3 = vec3<f32>;
type float4 = vec4<f32>;


struct Params {
    width: u32;
    height: u32;
    frame: u32;
};

struct StorageBuffer {
    data: array<atomic<uint>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var col: texture_storage_2d<rgba16float,write>;
[[group(0), binding(2)]] var<storage,read_write> buf: StorageBuffer;
[[group(0), binding(3)]] var tex: texture_2d_array<float>;
[[group(0), binding(4)]] var texs: texture_storage_2d_array<rgba16float,write>;
[[group(0), binding(5)]] var nearest: sampler;
[[group(0), binding(6)]] var bilinear: sampler;

fn hash44(p: float4) -> float4 {
	var p4 = fract(p * float4(.1031, .1030, .0973, .1099));
    p4 = p4 + dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

let dt = 1.;
let n = float2(0., 1.);
let e = float2(1., 0.);
let s = float2(0., -1.);
let w = float2(-1., 0.);

fn A(fragCoord: float2) -> float4 {
    let resolution = float2(float(params.width), float(params.height));
    return textureSampleLevel(tex, nearest, fract(fragCoord / resolution), 0, 0.);
}

fn A(fragCoord: uint2) -> float4 {
    let resolution = float2(float(params.width), float(params.height));
    return textureSampleLevel(tex, nearest, fract(fragCoord / resolution), 0, 0.);
}

fn B(fragCoord: float2) -> float4 {
    let resolution = float2(float(params.width), float(params.height));
    return textureSampleLevel(tex, bilinear, fract(fragCoord / resolution), 1, 0.);
}

fn T(fragCoord: float2) -> float4 {
    return B(fragCoord - dt * B(fragCoord).xy);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_velocity([[builtin(global_invocation_id)]] global_id: uint3) {
    let u = float2(global_id.xy) + 0.5;
    var r = T(u);
    r.x = r.x - dt * 0.25 * (T(u+e).z - T(u+w).z);
    r.y = r.y - dt * 0.25 * (T(u+n).z - T(u+s).z);

    if (params.frame < 3u) { r = float4(0.); }
    textureStore(texs, int2(global_id.xy), 0, r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_pressure([[builtin(global_invocation_id)]] global_id:  uint3) {
    let resolution = float2(float(params.width), float(params.height));
    let u = float2(global_id.xy) + 0.5;
    var r = A(u);
    r.z = r.z - dt * 0.25 * (A(u+e).x - A(u+w).x + A(u+n).y - A(u+s).y);

    let t = float(params.frame) / 120.;
    let o = resolution/2. * (1. + .75 * float2(cos(t/15.), sin(2.7*t/15.)));
    r = mix(r, float4(0.5 * sin(dt * 2. * t) * sin(dt * t), 0., r.z, 1.), exp(-0.2 * length(u - o)));
    textureStore(texs, int2(global_id.xy), 1, r);
}


fn insertion_sort(i: ptr<function,uint4>, d: ptr<function,float4>, uint i_, float d_) {
    if(any(uint4(i_,i_,i_,i_) == i)) return;
    if     (d_ < d[0])
        i = uint4(i_,i.xyz),    d = float4(d_,d.xyz);
    else if(d_ < d[1])
        i = uint4(i.x,i_,i.yz), d = float4(d.x,d_,d.yz);
    else if(d_ < d[2])
        i = uint4(i.xy,i_,i.z), d = float4(d.xy,d_,d.z);
    else if(d_ < d[3])
        i = uint4(i.xyz,i_),    d = float4(d.xyz,d_);
}

fn id_to_pos(id: uint) -> uint2 {
    return uint2(id % params.width, id / params.width);
}

fn pos_to_id(pos: uint3) -> uint {
    return pos.x + pos.y * params.width;
}

fn get_distance_with_nulls(id: uint, pos0: float2, pos1: float2) {
    // return arbitrarily large number for null id
    return select(float(1.0e12), distance(pos0, pos1), id >= 0);
}

fn get_distance_vec(fragCoord: uint2, ids: uint4) -> float4 {
    let pos0 = A(id_to_pos(ids.x)).xy;
    let pos1 = A(id_to_pos(ids.y)).xy;
    let pos2 = A(id_to_pos(ids.z)).xy;
    let pos3 = A(id_to_pos(ids.w)).xy;
    let ffC = float2(fragCoord);
    return float4(get_distance_with_nulls(ids.x, pos0, ffC),
                  get_distance_with_nulls(ids.y, pos1, ffC),
                  get_distance_with_nulls(ids.z, pos2, ffC),
                  get_distance_with_nulls(ids.w, pos3, ffC));
}

fn get_uint4_atomic(id: uint) -> uint4 {
    let i0 = atomicLoad(&buf.data[id*4u+0u]);
    let i1 = atomicLoad(&buf.data[id*4u+1u]);
    let i2 = atomicLoad(&buf.data[id*4u+2u]);
    let i3 = atomicLoad(&buf.data[id*4u+3u]);
    return uint4(i0,i1,i2,i3);
}

fn set_uint4_atomic(id: uint, val: uint4) {
    atomicStore(&buf.data[id*4u+0u], val.x);
    atomicStore(&buf.data[id*4u+1u], val.y);
    atomicStore(&buf.data[id*4u+2u], val.z);
    atomicStore(&buf.data[id*4u+3u], val.w);
}

fn load_sort_atomic(id_here: uint, pos_here: uint2, candidate_id: uint, candidate_pos: float2) {
    let id_here = pos_to_id(pos_here);
    let candidate_id = pos_to_id(candidate);
    let candidate_dist = distance(float2(pos_here), candidate_pos);
    let ids = get_uint4_atomic(id_here);

    float4 dists = get_distance_vec(pos_here, ids);
    insertion_sort(&ids, &dists, candidate_id, candidate_dist);
    set_uint4_atomic(id_here, ids);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_caustics([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let resolution = float2(float(params.width), float(params.height));
    for (var i = 0; i < 25; i = i+1) {
        let h = hash44(float4(float2(global_id.xy), float(params.frame), float(i)));
        var p = float2(global_id.xy) + h.xy;
        let z = mix(.3, 1., h.z);
        let c = max(cos(z*6.2+float4(1.,2.,3.,4.)),float4(0.));
        let n = A(p + float2(0., 1.));
        let e = A(p + float2(1., 0.));
        let s = A(p - float2(0., 1.));
        let w = A(p - float2(1., 0.));
        let grad = 0.25 * float2(e.z - w.z, n.z - s.z);
        p = p + 1e5 * grad * z;
        p = fract(p / resolution) * resolution;
        let id = u32(p.x) + u32(p.y) * params.width;
        atomicAdd(&buf.data[id*4u+0u], u32(c.x * 256.));
        atomicAdd(&buf.data[id*4u+1u], u32(c.y * 256.));
        atomicAdd(&buf.data[id*4u+2u], u32(c.z * 256.));
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_image([[builtin(global_invocation_id)]] global_id: uint3) {
    let id = global_id.x + global_id.y * params.width;
    let x = float(atomicLoad(&buf.data[id*4u+0u]));
    let y = float(atomicLoad(&buf.data[id*4u+1u]));
    let z = float(atomicLoad(&buf.data[id*4u+2u]));
    var r = float3(x, y, z) / 256.;
    r = r * sqrt(r) / 5e3;
    r = r * float3(.5, .75, 1.);
    textureStore(col, int2(global_id.xy), float4(r, 1.));
    atomicStore(&buf.data[id*4u+0u], u32(x * .9));
    atomicStore(&buf.data[id*4u+1u], u32(y * .9));
    atomicStore(&buf.data[id*4u+2u], u32(z * .9));
}
