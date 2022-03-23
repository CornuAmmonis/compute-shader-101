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
    data: array<atomic<u32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var col: texture_storage_2d<rgba32float,write>;
[[group(0), binding(2)]] var<storage,read_write> buf: StorageBuffer;
[[group(0), binding(3)]] var tex: texture_2d_array<f32>;
[[group(0), binding(4)]] var texs: texture_storage_2d_array<rgba32float,write>;
[[group(0), binding(5)]] var nearest: sampler;
[[group(0), binding(6)]] var bilinear: sampler;

fn pcg(p: uint4) -> uint4 {
	var v = p * 1664525u + 1013904223u;

    // no += allowed here
	v.x = v.x + v.y * v.w;
	v.y = v.y + v.z * v.x;
	v.z = v.z + v.x * v.y;
	v.w = v.w + v.y * v.z;

	v.x = v.x + v.y * v.w;
	v.y = v.y + v.z * v.x;
	v.z = v.z + v.x * v.y;
	v.w = v.w + v.y * v.z;

	v = v ^ (v >> uint4(16u));

	return v;
}

fn hash44(p: float4) -> float4 {
	var p4 = fract(p * float4(.1031, .1030, .0973, .1099));
    p4 = p4 + dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

var<private> seed: float4;

fn random() -> float4 {
	seed = float4(pcg(bitcast<uint4>(seed))) / float(uint(0xffffffffu));
	//seed = hash44(seed);
	return seed;
}

fn set_seed(global_id: uint3) {
    seed = float4(global_id.xyxy + params.frame);// + float4(
            //bitcast<float>(0xbce570e6u),
            //bitcast<float>(0xc765e2a9u),
            //bitcast<float>(0x1701f5dcu),
            //bitcast<float>(0xa46c411cu)
           //);
}


let n = float2(0., 1.);
let e = float2(1., 0.);
let s = float2(0., -1.);
let w = float2(-1., 0.);

fn A_f2(fragCoord: float2) -> float4 {
    let resolution = float2(float(params.width), float(params.height));
    return textureSampleLevel(tex, nearest, fract(fragCoord / resolution), 0, 0.);
}

fn A_u2(fragCoord: uint2) -> float4 {
    return textureLoad(tex, int2(fragCoord), 0, 0);
}

fn B(fragCoord: float2) -> float4 {
    let resolution = float2(float(params.width), float(params.height));
    return textureSampleLevel(tex, bilinear, fract(fragCoord / resolution), 1, 0.);
}

fn B_u2(fragCoord: uint2) -> float4 {
    return textureLoad(tex,  int2(fragCoord), 1, 0);
}

fn eucMod_i(a: int, b: int) -> int {
    return a - abs(b) * (a / abs(b));
}

fn eucMod_f(a: float, b: float) -> float {
    return a - abs(b) * floor(a / abs(b));
}

fn eucMod_i2(a: int2, b: int2) -> int2 {
    return int2(eucMod_i(a.x,b.x), eucMod_i(a.y,b.y));
}

fn wrap_to_bounds(pos: int2) -> uint2 {
    let bounds = int2(int(params.width), int(params.height));
    return uint2(eucMod_i2(pos, bounds));
}

fn id_to_pos(id: uint) -> uint2 {
    return uint2(id % params.width, id / params.width);
}

fn pos_to_id_u2(pos: uint2) -> uint {
    return pos.x + pos.y * params.width;
}

fn pos_to_id_u3(pos: uint2) -> uint {
    return pos.x + pos.y * params.width;
}

fn particle_to_pos_u2(particle_data: float4) -> uint2 {
    return wrap_to_bounds(int2(round(particle_data.xy)));
}

fn insertion_sort(i: ptr<function,uint4>, d: ptr<function,float4>, i_: uint, d_: float) {
    if(any(uint4(i_) == *i)) { return; }
    if (d_ < (*d).x) {
        *i = uint4(i_,(*i).xyz);
        *d = float4(d_,(*d).xyz);
    } else if(d_ < (*d).y) {
        *i = uint4((*i).x,i_,(*i).yz);
        *d = float4((*d).x,d_,(*d).yz);
    } else if(d_ < (*d).z) {
        *i = uint4((*i).xy,i_,(*i).z);
        *d = float4((*d).xy,d_,(*d).z);
    } else if(d_ < (*d).w) {
        *i = uint4((*i).xyz,i_);
        *d = float4((*d).xyz,d_);
    }
}

let ARBITRARILY_LARGE_DISTANCE = 1.0e6;
let INVALID_ID = 0xFFFFFFFFu;

fn get_distance_with_nulls(id: uint, pos0: float2, pos1: float2) -> float {
    // return arbitrarily large number for null id
    return select(ARBITRARILY_LARGE_DISTANCE, distance(pos0, pos1), id != INVALID_ID);
}

fn get_distance_vec(fragCoord: uint2, ids: uint4) -> float4 {
    let pos0 = A_u2(id_to_pos(ids.x)).xy;
    let pos1 = A_u2(id_to_pos(ids.y)).xy;
    let pos2 = A_u2(id_to_pos(ids.z)).xy;
    let pos3 = A_u2(id_to_pos(ids.w)).xy;
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

var<private> nearest_ids: array<uint,20>;

fn set_nearest_arr_at_offset(offset: uint, vals: uint4) {
    nearest_ids[offset + 0u] = vals.x;
    nearest_ids[offset + 1u] = vals.y;
    nearest_ids[offset + 2u] = vals.z;
    nearest_ids[offset + 3u] = vals.w;
}

fn add_offset(pos: uint2, offset: int2) -> uint2 {
    return wrap_to_bounds(int2(pos) + offset);
}

fn populate_nearest(pos_here: uint2) {
    let n0 = get_uint4_atomic(pos_to_id_u2(pos_here));
    let n1 = get_uint4_atomic(pos_to_id_u2(add_offset(pos_here,int2(-1, 0))));
    let n2 = get_uint4_atomic(pos_to_id_u2(add_offset(pos_here,int2( 1, 0))));
    let n3 = get_uint4_atomic(pos_to_id_u2(add_offset(pos_here,int2( 0,-1))));
    let n4 = get_uint4_atomic(pos_to_id_u2(add_offset(pos_here,int2( 0, 1))));
    set_nearest_arr_at_offset(0u,n0);
    set_nearest_arr_at_offset(4u,n1);
    set_nearest_arr_at_offset(8u,n2);
    set_nearest_arr_at_offset(12u,n3);
    set_nearest_arr_at_offset(16u,n4);
}

fn get_pressure_diff(pos: uint2) -> float2 {
    let B_w = B_u2(add_offset(pos.xy,int2(-1, 0))).x;
    let B_e = B_u2(add_offset(pos.xy,int2( 1, 0))).x;
    let B_s = B_u2(add_offset(pos.xy,int2( 0,-1))).x;
    let B_n = B_u2(add_offset(pos.xy,int2( 0, 1))).x;
    return float2(B_e - B_w, B_n - B_s);
}

fn distance_kernel(distance: float) -> float {
    return exp(-0.3*distance);
}

fn get_weighted_velocity_avg(particle_data: float4) -> float2 {
    let particle_pos = particle_to_pos_u2(particle_data);
    let particle_local_id = pos_to_id_u2(particle_pos);
    let particle_nearest_ids = get_uint4_atomic(particle_local_id);

    var weight_sum = 0.;
    var velocity_sum = float2(0.);

    for (var i = 0; i < 4; i = i + 1) {
        let neighbor_particle_data = A_u2(id_to_pos(particle_nearest_ids[i]));
        let neighbor_particle_vec = neighbor_particle_data.xy - particle_data.xy;
        let neighbor_particle_dist = length(neighbor_particle_vec);
        let kernel_weight = distance_kernel(neighbor_particle_dist);
        velocity_sum = velocity_sum + kernel_weight * neighbor_particle_data.zw;
        weight_sum = weight_sum + kernel_weight;
    }

    return select(float2(0.), velocity_sum / weight_sum, weight_sum > 0.);
}

fn get_separation_velocity(particle_data: float4) -> float2 {
    let particle_pos = particle_to_pos_u2(particle_data);
    let particle_local_id = pos_to_id_u2(particle_pos);
    let particle_nearest_ids = get_uint4_atomic(particle_local_id);

    var weight_sum = 0.;
    var velocity_sum = float2(0.);

    for (var i = 0; i < 4; i = i + 1) {
        let neighbor_particle_data = A_u2(id_to_pos(particle_nearest_ids[i]));
        let neighbor_particle_vec = neighbor_particle_data.xy - particle_data.xy;
        let neighbor_particle_dist = length(neighbor_particle_vec);
        let kernel_weight = distance_kernel(neighbor_particle_dist);
        velocity_sum = velocity_sum - kernel_weight * neighbor_particle_vec;
        weight_sum = weight_sum + kernel_weight;
    }

    return select(float2(0.), velocity_sum / weight_sum, weight_sum > 0.);
}

let velocity_dt = 0.05;

[[stage(compute), workgroup_size(16, 16)]]
fn main_velocity([[builtin(global_invocation_id)]] global_id: uint3) {
    set_seed(global_id);

    let state = A_u2(global_id.xy);
    let velocity_avg = get_weighted_velocity_avg(state);
    let separation_velocity = get_separation_velocity(state);
    let pressure_diff = get_pressure_diff(global_id.xy);
    let new_velocity = mix(state.zw, velocity_avg - pressure_diff + separation_velocity, velocity_dt);
    var new_state = float4(state.xy + velocity_dt * new_velocity, new_velocity);

    new_state.z = select(new_state.z, -new_state.z, (state.x < 0. && state.z < 0.) || (state.x > float(params.width) && state.z > 0.));
    new_state.w = select(new_state.w, -new_state.w, (state.y < 0. && state.w < 0.) || (state.y > float(params.height) && state.w > 0.));
    new_state.x = clamp(new_state.x, -1., float(params.width));
    new_state.y = clamp(new_state.y, -1., float(params.height));

    if (params.frame < 3u) {
        let f_size = float2(float(params.width), float(params.height));
        let rng2 = (random() + random() + random() + random()) / 4.;
        new_state = (rng2.xyxy - float4(0.,0.,.5,.5)) * float4(f_size.xy,10.,10.);
    }
    textureStore(texs, int2(global_id.xy), 0, new_state);
}

let pressure_dt = 0.5;
let pressure_decay = 0.95;
[[stage(compute), workgroup_size(16, 16)]]
fn main_pressure([[builtin(global_invocation_id)]] global_id:  uint3) {
    set_seed(global_id);
    let particle_data = A_u2(global_id.xy);
    let particle_pos = particle_to_pos_u2(particle_data);
    let particle_local_id = pos_to_id_u2(particle_pos);
    let particle_nearest_ids = get_uint4_atomic(particle_local_id);
    let particle_nearest_dists = get_distance_vec(particle_pos, particle_nearest_ids);

    let pressure_prev = B_u2(global_id.xy);
    var pressure_delta = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let neighbor_particle_data = A_u2(id_to_pos(particle_nearest_ids[i]));
        let neighbor_particle_vec = neighbor_particle_data.xy - particle_data.xy;
        let neighbor_particle_dist = length(neighbor_particle_vec);
        let partial_pressure = distance_kernel(neighbor_particle_dist) * dot(-neighbor_particle_vec, neighbor_particle_data.zw);
        pressure_delta = pressure_delta + partial_pressure;
    }
    let new_pressure = pressure_prev * pressure_decay + pressure_dt * pressure_delta;
    textureStore(texs, int2(global_id.xy), 1, new_pressure);
}

let POPULATE_FROM_GLOBAL_NEIGHBORS = true;

[[stage(compute), workgroup_size(16, 16)]]
fn main_particle_spray([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    set_seed(global_id);
    let resolution = float2(float(params.width), float(params.height));
    let global_pos = uint2(global_id.xy);
    let global_id_unrolled = pos_to_id_u2(global_pos);

    let particle_data = A_u2(global_pos);
    let particle_pos = particle_to_pos_u2(particle_data);
    let particle_global_id = pos_to_id_u2(global_pos);
    let particle_local_id = pos_to_id_u2(particle_pos);
    var particle_nearest_ids = get_uint4_atomic(particle_local_id);
    var particle_nearest_dists = get_distance_vec(particle_pos, particle_nearest_ids);
    let particle_dist = distance(float2(particle_pos), particle_data.xy); //won't work for wrapped bounds


    var particle_nearest_ids_init = uint4(INVALID_ID);
    var particle_nearest_dists_init = float4(ARBITRARILY_LARGE_DISTANCE);
    // resort the existing index
    for (var i = 0; i < 4; i = i+1) {
        insertion_sort(&particle_nearest_ids_init, &particle_nearest_dists_init, particle_nearest_ids[i], particle_nearest_dists[i]);
    }
    // insert the new particle
    insertion_sort(&particle_nearest_ids_init, &particle_nearest_dists_init, particle_global_id, particle_dist);

    // set result
    set_uint4_atomic(particle_local_id, particle_nearest_ids_init);

    //populate_nearest(particle_pos_here);
    //
    //for (var i = 0; i < 20; i = i+1) {
    //    let candidate_id = nearest_ids[i];
    //    let candidate_pos = A_u2(id_to_pos(candidate_id)).xy;
    //    let candidate_dist = distance(float2(particle_pos_here), candidate_pos);
    //    insertion_sort(&ids, &dists, candidate_id, candidate_dist);
    //}
    //set_uint4_atomic(particle_id_here, ids);
    //

    if (POPULATE_FROM_GLOBAL_NEIGHBORS) {
        var global_ids = get_uint4_atomic(global_id_unrolled);
        var global_dists = get_distance_vec(global_pos, global_ids);
        populate_nearest(global_pos);

        for (var i = 0; i < 20; i = i+1) {
            let candidate_id = nearest_ids[i];
            let candidate_pos = A_u2(id_to_pos(candidate_id)).xy;
            let candidate_dist = distance(float2(global_pos), candidate_pos);
            insertion_sort(&global_ids, &global_dists, candidate_id, candidate_dist);
        }
        set_uint4_atomic(global_id_unrolled, global_ids);
    }

}

[[stage(compute), workgroup_size(16, 16)]]
fn main_image([[builtin(global_invocation_id)]] global_id: uint3) {
    set_seed(global_id);
    let id = global_id.x + global_id.y * params.width;
    let ids = get_uint4_atomic(id);
    let dists = get_distance_vec(uint2(global_id.xy), ids);
    let r = 1.0 / (4.0 + 20.*dists*dists);
    //let r = 1.0 / (1.0 + 0.5*dists);
    let c = dot(r, float4(1.));
    //let c = r.xxxx;
    //let resolution = float2(float(params.width), float(params.height));
    //let c = A_u2(global_id.xy) / float4(resolution, resolution);

    textureStore(col, int2(global_id.xy), float4(c));
}
