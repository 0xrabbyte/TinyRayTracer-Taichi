import taichi as ti
from taichi.lang import kernel_profiler_clear
from taichi.lang.ops import append

ti.init(arch=ti.gpu)

TEXTURE_MAXX = 2048
TEXTURE_MAXY = 2048
TEXTURE_N = 10

M = 30000
Pi = 3.141592653
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
samples_per_pixel = 4
max_depth = 10
sample_on_unit_sphere_surface = True

look_alpha = 10.0 * Pi / 180.0
lookfrom = ti.Vector.field(3, float, shape = ())
lookat = ti.Vector.field(3, float, shape = ())
vup = ti.Vector.field(3, float, shape = ())
lookfrom[None] = ti.Vector([0.0, 1.0, -5.0])
lookat[None] = ti.Vector([0.0, 1.0, -1.0])
vup[None] = ti.Vector([0.0, 1.0, 0.0])
vfov = ti.field(float, shape = ())
vfov[None] = 60.0
look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
dist_to_focus = ti.field(float, shape = ())
dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
aperture = ti.field(float, shape = ())
aperture[None] = 0.1
cam_time = ti.Vector.field(2, float, shape=())
cam_time[None] = ti.Vector([0.0, 1.0])

@ti.func
def random_double(a, b):
    return a + ti.random() * (b - a)

@ti.func
def random_in_unit_sphere(): # Here is the optimization
    theta = 2.0 * Pi * ti.random()
    phi = ti.acos((2.0 * ti.random()) - 1.0)
    r = ti.pow(ti.random(), 1.0/3.0)
    return ti.Vector([r * ti.sin(phi) * ti.cos(theta), r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi)])

@ti.func
def random_unit_vector():
    theta = 2.0 * Pi * ti.random()
    phi = ti.acos((2.0 * ti.random()) - 1.0)
    return ti.Vector([ti.sin(phi) * ti.cos(theta), ti.sin(phi) * ti.sin(theta), ti.cos(phi)])    

@ti.func
def random_in_hemisphere(normal):
    in_unit_sphere = random_in_unit_sphere()
    if (in_unit_sphere.dot(normal) < 0.0):
        in_unit_sphere *= -1
    return in_unit_sphere

@ti.func
def random_in_unit_disk():
    theta = 2.0 * Pi * ti.random()
    return ti.Vector([ti.cos(theta), ti.sin(theta)])

@ti.data_oriented
class ray:
    def __init__(self, origin, direction, tm = 0.0):
        self.orig = origin
        self.dir = direction
        self.tm = tm
    def origin(self):
        return self.orig
    def direction(self):
        return self.dir
    def time(self):
        return self.tm
    def at(self, t):
        return self.orig + t * self.dir

@ti.data_oriented
class camera:
    def __init__(self):
        self.viewport_height = ti.field(float, shape = ())
        self.viewport_width = ti.field(float, shape = ())
        self.horizontal = ti.Vector.field(3, float, shape = ())
        self.vertical = ti.Vector.field(3, float, shape = ())
        self.origin = ti.Vector.field(3, float, shape = ())
        self.lower_left_corner = ti.Vector.field(3, float, shape = ())
        self.focal_length = ti.field(float, shape = ())
        self.lens_radius = ti.field(float, shape = ())
        self.time0 = ti.field(float, shape = ())
        self.time1 = ti.field(float, shape = ())
        self.reset_view()
    @ti.kernel
    def reset_view(self):
        theta = vfov[None] * Pi / 180.0
        h = ti.tan(theta / 2)
        self.viewport_height[None] = 2.0 * h
        self.viewport_width[None] = aspect_ratio * self.viewport_height[None]

        w = (lookfrom[None] - lookat[None]).normalized()
        u = (vup[None].cross(w)).normalized()
        v = w.cross(u)

        self.focal_length[None] = 1.0
        self.horizontal[None] = dist_to_focus[None] * self.viewport_width[None] * u
        self.vertical[None] = dist_to_focus[None] * self.viewport_height[None] * v
        self.origin[None] = ti.Vector([lookfrom[None][0], lookfrom[None][1], lookfrom[None][2]])
        self.lower_left_corner[None] = self.origin[None] - self.horizontal[None] / 2 - self.vertical[None] / 2 - dist_to_focus[None] * w
        self.lens_radius[None] = aperture[None] / 2.0

        self.time0[None] = cam_time[None][0]
        self.time1[None] = cam_time[None][1]
    @ti.func
    def get_ray(self, u, v):
        rd = self.lens_radius[None] * random_in_unit_disk()
        offset = u * rd[0] + v * rd[1]
        return ray(self.origin[None] + offset, self.lower_left_corner[None] + u * self.horizontal[None] + v * self.vertical[None] - self.origin[None] - offset, random_double(self.time0[None], self.time1[None]))

@ti.func
def set_face_normal(r, outward_normal):
    front_face = (r.direction().dot(outward_normal) < 0)
    normal = outward_normal
    if (not front_face):
        normal = -outward_normal
    return front_face, normal

@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(1.0 - r_out_perp.dot(r_out_perp)) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

material_cnt = ti.field(ti.i32, shape=())
material_cnt[None] = 0
material_kind = ti.field(ti.i32)
material_albedo = ti.Vector.field(3, float)
material_extra = ti.field(float)
ti.root.dense(ti.i, 5 * M).place(material_kind, material_albedo, material_extra)

@ti.data_oriented
class materials:
    def __init__(self, s, a, e):
        self.kind = s
        self.albedo = a
        self.extra = e #Fuzz/Refraction_Rate

def materials_add(k, a, e):
    material_kind[material_cnt[None]] = k
    material_albedo[material_cnt[None]] = a
    material_extra[material_cnt[None]] = e
    material_cnt[None] += 1
    return material_cnt[None] - 1

@ti.func
def scatter(k, r, p, normal, ff, u, v):
    r_direction = normal
    reached = True
    if (material_kind[k] == 0 or material_kind[k] == 4 or material_kind[k] == 5 or material_kind[k] == 6): # lambertian
        if sample_on_unit_sphere_surface: r_direction = normal + random_unit_vector()
        else: r_direction = normal + random_in_unit_sphere()
    if (material_kind[k] == 1): # metal
        r_direction = reflect((r.direction()).normalized(), normal)
        if sample_on_unit_sphere_surface: r_direction += material_extra[k] * random_unit_vector()
        else: r_direction += material_extra[k] * random_in_unit_sphere()
        reached = (ray(p, r_direction, r.time()).direction().dot(normal) > 0)
    if (material_kind[k] == 2): # dielectric
        refraction_ratio = material_extra[k]
        if (ff):
            refraction_ratio = 1.0 / material_extra[k]
        cos_theta = min(normal.dot(-(r.direction()).normalized()), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
        if (refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random()):
            r_direction = reflect((r.direction()).normalized(), normal)
        else:
            r_direction = refract((r.direction()).normalized(), normal, refraction_ratio)
    if (material_kind[k] == 3): # diffuse_light
        reached = False
    return reached, p, r_direction, r.time()

@ti.pyfunc
def perlin_noise(p):
    u, v, w = p[0] - int(p[0]), p[1] - int(p[1]), p[2] - int(p[2])
    i, j, k = int(p[0]), int(p[1]), int(p[2])
    uu, vv, ww = u*u*(3-2*u), v*v*(3-2*v), w*w*(3-2*w)
    accum = 0.0
    for di, dj, dk in ti.ndrange(2, 2, 2):
        trilerp = (di * uu + (1-di)*(1-uu)) * (dj*vv + (1-dj)*(1-vv)) * (dk*ww + (1-dk)*(1-ww))
        c = perlin_noise_vec[perlin_noise_perm_x[(i + di) & 255] ^ perlin_noise_perm_y[(j + dj) & 255] ^ perlin_noise_perm_z[(k + dk) & 255]]
        accum += trilerp * ti.Vector([u - di, v - di, w - di]).dot(c)
    accum = accum * 0.5 + 1.0
    return accum

textures = ti.Vector.field(3, float, shape = (TEXTURE_N, TEXTURE_MAXX, TEXTURE_MAXY))
textures_cnt = ti.field(ti.i32, shape = ())
textures_cnt[None] = 0
textures_size = ti.Vector.field(2, ti.i32, shape = (TEXTURE_N))

@ti.func
def get_albedo(k, me, al, u, v, p):
    if (k == 4):
        if (ti.sin(10.0 * p[0]) * ti.sin(10.0 * p[1]) * ti.sin(10.0 * p[2]) > 0.0):
            al = ti.Vector([0.9, 0.9, 0.9])
    if (k==5):
        accum = 0.0
        tmp_p = p
        w = 1.0
        for i in range(7):
            accum += w * perlin_noise(tmp_p)
            w *= 0.5
            tmp_p *= 2
        al *= (0.4 * ti.abs(accum) + 1.0 + ti.sin(0.1 * p[2])) * 0.5
    if (k==6):
        kk = int(me)
        width, height = textures_size[kk][0], textures_size[kk][1]
        i, j = int(u * width), int(v * height)
        if (i < 0): i = 0
        if (j < 0): j = 0
        if (i >= width): i = width - 1
        if (j >= height):j = height- 1
        al *= textures[kk, i, j] / 255.0
    return al

perlin_point_count = 256
perlin_noise_perm_x = ti.field(ti.i32)
perlin_noise_perm_y = ti.field(ti.i32)
perlin_noise_perm_z = ti.field(ti.i32)
perlin_noise_vec = ti.Vector.field(3, float, shape=(perlin_point_count))
ti.root.dense(ti.i, perlin_point_count).place(perlin_noise_perm_x, perlin_noise_perm_y, perlin_noise_perm_z)

@ti.kernel
def gen_perlin_noise():
    for i in range(perlin_point_count):
        perlin_noise_vec[i] = ti.Vector([ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0])
        perlin_noise_perm_x[i], perlin_noise_perm_y[i], perlin_noise_perm_z[i] = i, i, i
    for i in range(perlin_point_count):
        p = int(ti.random() * perlin_point_count)
        perlin_noise_perm_x[i], perlin_noise_perm_x[p] = perlin_noise_perm_x[p], perlin_noise_perm_x[i]
    for i in range(perlin_point_count):
        p = int(ti.random() * perlin_point_count)
        perlin_noise_perm_y[i], perlin_noise_perm_y[p] = perlin_noise_perm_y[p], perlin_noise_perm_y[i]
    for i in range(perlin_point_count):
        p = int(ti.random() * perlin_point_count)
        perlin_noise_perm_z[i], perlin_noise_perm_z[p] = perlin_noise_perm_z[p], perlin_noise_perm_z[i]

objs_type = ti.field(ti.i32)
objs_ind = ti.field(ti.i32)
ti.root.dense(ti.i, 5 * M).place(objs_type, objs_ind)
objs_cnt = ti.field(ti.i32, shape = ())
pur_objs_cnt = ti.field(ti.i32, shape = ())
objs_cnt[None] = 0
pur_objs_cnt[None] = 0

def surrounding_box(box0, box1):
    return AABB(ti.Vector([ti.min(box0.min()[0], box1.min()[0]), ti.min(box0.min()[1], box1.min()[1]), ti.min(box0.min()[2], box1.min()[2])])
                ,ti.Vector([ti.max(box0.max()[0], box1.max()[0]), ti.max(box0.max()[1], box1.max()[1]), ti.max(box0.max()[2], box1.max()[2])]))

def get_bounding_box(id, t0, t1):
    if (objs_type[id] == 0):
        return True, AABB(AABBs_min[objs_ind[id]], AABBs_max[objs_ind[id]])
    elif(objs_type[id] == 1):
        return sphere(sphere_centers[objs_ind[id]], sphere_radius[objs_ind[id]], sphere_material[objs_ind[id]]).bounding_box(t0, t1)
    elif(objs_type[id] == 2):
        tmp_moving_sphere = moving_sphere(moving_sphere_center0s[objs_ind[id]], moving_sphere_center1s[objs_ind[id]],
                        moving_sphere_time0[objs_ind[id]], moving_sphere_time1[objs_ind[id]],
                        moving_sphere_radius[objs_ind[id]], moving_sphere_material[objs_ind[id]])
        return tmp_moving_sphere.bounding_box(t0, t1)
    elif(objs_type[id] == 3):
        return tri_bounding(objs_ind[id], t0, t1)

@ti.func
def obj_hit(id, r, tmin, tmax):
    t = 1.0
    p = ti.Vector([0.0, 0.0, 0.0])
    front_face = False
    normal = ti.Vector([0.0, 0.0, 0.0])
    u = 1.0
    v = 1.0
    m = 0
    if (objs_type[id] == 0):
        if AABB_hit(objs_ind[id], r, tmin, tmax):
            t = 1.0
        else:
            t = -1.0
    elif(objs_type[id] == 1):
        t, p, front_face, normal, u, v, m = sphere_hit(objs_ind[id], r, tmin, tmax)
    elif(objs_type[id] == 2):
        t, p, front_face, normal, u, v, m = moving_sphere_hit(objs_ind[id], r, tmin, tmax)
    elif(objs_type[id] == 3):
        t, p, front_face, normal, u, v, m = tri_hit(objs_ind[id], r, tmin, tmax)
    return t, p, front_face, normal, u, v, m


AABBs_min = ti.Vector.field(3, float)
AABBs_max = ti.Vector.field(3, float)
ti.root.dense(ti.i, M).place(AABBs_min, AABBs_max)
AABBs_cnt = ti.field(ti.i32, shape=())
AABBs_cnt[None] = 0

@ti.data_oriented
class AABB:
    def __init__(self, a, b):
        self._min = a
        self._max = b
    @ti.pyfunc
    def min(self):return self._min
    @ti.pyfunc
    def max(self):return self._max
    @ti.pyfunc
    def bounding_box(self, t0, t1):
        return True, self
@ti.func
def AABB_hit(id, r, tmin, tmax):
    for i in ti.static(range(3)):
        invD = 1.0 / r.direction()[i]
        t0 = (AABBs_min[id][i] - r.origin()[i]) * invD
        t1 = (AABBs_max[id][i] - r.origin()[i]) * invD
        if (invD < 0.0):
            t0, t1 = t1, t0
        tmin = ti.max(t0, tmin)
        tmax = ti.min(t1, tmax)
    return tmin < tmax

bvh_tree = []
nxt = ti.field(ti.i32)
lft = ti.field(ti.i32)
ti.root.dense(ti.i, 2 * M).place(nxt, lft)
objs_id = ti.field(ti.i32, shape = (5 * M))
def bvh_init(u, tnxt):
    nxt[u] = tnxt
    lft[u] = bvh_tree[u].left
    if bvh_tree[u].right > bvh_tree[u].left:
        bvh_init(bvh_tree[u].left, bvh_tree[u].right)
        bvh_init(bvh_tree[u].right, tnxt)
    elif bvh_tree[u].left > 0:
        bvh_init(bvh_tree[u].right, tnxt)

@ti.data_oriented
class bvh_node:
    def __init__(self, t0, t1):
        self.left = 0
        self.right = 0
        self.time0 = t0
        self.time1 = t1
    @ti.pyfunc
    def bounding_box(self, t0, t1):
        return True, self.box
@ti.func
def bvh_hit(r, t_min, t_max):
    closest_so_far = t_max
    t = -1.0
    p = ti.Vector([0.0, 0.0, 0.0])
    front_face = False
    normal = ti.Vector([0.0, 0.0, 0.0])
    m = 0
    u = 1.0
    v = 1.0
    qh = 0
    while qh != -1:
        tmp_t, tmp_p, tmp_front_face, tmp_normal, tu, tv, tm = obj_hit(objs_id[qh], r, t_min, closest_so_far)
        if lft[qh] == 0:
            if tmp_t > 0.0:
                closest_so_far = tmp_t
                t, p, front_face, normal, u, v, m = tmp_t, tmp_p, tmp_front_face, tmp_normal, tu, tv, tm
            qh = nxt[qh]
        else:
            if tmp_t > 0.0:
                qh = lft[qh]
            else:
                qh = nxt[qh]
    return t, p, front_face, normal, u, v, m

@ti.func
def bruteforce_hit(r, t_min, t_max):
    closest_so_far = t_max
    t = -1.0
    p = ti.Vector([0.0, 0.0, 0.0])
    front_face = False
    normal = ti.Vector([0.0, 0.0, 0.0])
    m = 0
    u = 1.0
    v = 1.0
    for i in range(pur_objs_cnt[None]):
        if (objs_type[i] > 0):
            tmp_t, tmp_p, tmp_front_face, tmp_normal, tu, tv, tm = obj_hit(i, r, t_min, closest_so_far)
            if tmp_t > 0.0:
                closest_so_far = tmp_t
                t, p, front_face, normal, u, v, m = tmp_t, tmp_p, tmp_front_face, tmp_normal, tu, tv, tm
    return t, p, front_face, normal, u, v, m

@ti.kernel
def gen_axis() -> ti.i32:
    rand_num = ti.random()
    axis = 0
    if rand_num * 3.0 > 1.0 : axis += 1
    if rand_num * 3.0 > 2.0 : axis += 1
    return axis

tmp_min = ti.Vector.field(3, float, shape=())
tmp_max = ti.Vector.field(3, float, shape=())
temp_min = ti.Vector.field(3, float, shape=())
temp_max = ti.Vector.field(3, float, shape=())
def build_bvh_tree(u, objs):
    def get_x(e):
        is_hit, box =  AABB(AABBs_min[e], AABBs_max[e]).bounding_box(0.0, 0.0)
        return box.min()[0]
    def get_y(e):
        is_hit, box =  AABB(AABBs_min[e], AABBs_max[e]).bounding_box(0.0, 0.0)
        return box.min()[1]
    def get_z(e):
        is_hit, box = AABB(AABBs_min[e], AABBs_max[e]).bounding_box(0.0, 0.0)
        return box.min()[2]
    axis = gen_axis()
    if len(objs) == 0: return
    if (len(objs) == 1):
        bvh_tree.append(bvh_node(bvh_tree[u].time0, bvh_tree[u].time1))
        bvh_tree[u].left = bvh_tree[u].right = len(bvh_tree) - 1
        objs_id[len(bvh_tree) - 1] = objs[0]
    else:
        if axis == 0: objs.sort(key = get_x)
        if axis == 1: objs.sort(key = get_y)
        if axis == 2: objs.sort(key = get_z)
        if (len(objs) == 2):
            bvh_tree.append(bvh_node(bvh_tree[u].time0, bvh_tree[u].time1))
            bvh_tree[u].left = len(bvh_tree) - 1
            objs_id[len(bvh_tree) - 1] = objs[0]
            bvh_tree.append(bvh_node(bvh_tree[u].time0, bvh_tree[u].time1))
            bvh_tree[u].right = len(bvh_tree) - 1
            objs_id[len(bvh_tree) - 1] = objs[1]
        else:
            mid = (len(objs) + 1) // 2
            bvh_tree.append(bvh_node(bvh_tree[u].time0, bvh_tree[u].time1))
            bvh_tree[u].left = len(bvh_tree) - 1
            build_bvh_tree(bvh_tree[u].left, objs[:mid])
            bvh_tree.append(bvh_node(bvh_tree[u].time0, bvh_tree[u].time1))
            bvh_tree[u].right = len(bvh_tree) - 1
            build_bvh_tree(bvh_tree[u].right, objs[mid:])
    is_hit, box_a = get_bounding_box(objs_id[bvh_tree[u].left], bvh_tree[u].time0, bvh_tree[u].time1)
    temp_min[None], temp_max[None] = box_a.min(), box_a.max()
    is_hit, box_b = get_bounding_box(objs_id[bvh_tree[u].right], bvh_tree[u].time0, bvh_tree[u].time1)
    tmp_box = surrounding_box(AABB(temp_min[None], temp_max[None]), box_b)
    AABBs_min[AABBs_cnt[None]] = tmp_box.min()
    AABBs_max[AABBs_cnt[None]] = tmp_box.max()
    objs_type[objs_cnt[None]] = 0
    objs_ind[objs_cnt[None]] = AABBs_cnt[None]
    objs_id[u] = objs_cnt[None]
    objs_cnt[None] += 1
    AABBs_cnt[None] += 1

@ti.data_oriented
class hittable_list:
    def __init__(self):
        self.objects = []
    def add(self, type, object):
        if type == 1:
            sphere_centers[sphere_cnt[None]] = object.center
            sphere_radius[sphere_cnt[None]] = object.radius
            sphere_material[sphere_cnt[None]] = object.material
            objs_ind[objs_cnt[None]] = sphere_cnt[None]
            sphere_cnt[None] +=1
        if type == 2:
            moving_sphere_center0s[moving_sphere_cnt[None]] = object.center0
            moving_sphere_center1s[moving_sphere_cnt[None]] = object.center1
            moving_sphere_time0[moving_sphere_cnt[None]] = object.time0
            moving_sphere_time1[moving_sphere_cnt[None]] = object.time1
            moving_sphere_radius[moving_sphere_cnt[None]] = object.radius
            moving_sphere_material[moving_sphere_cnt[None]] = object.material
            objs_ind[objs_cnt[None]] = moving_sphere_cnt[None]
            moving_sphere_cnt[None] +=1
        if type == 3:
            tri_mesh_ind[tri_cnt[None]][0] = object.Aid
            tri_mesh_ind[tri_cnt[None]][1] = object.Bid
            tri_mesh_ind[tri_cnt[None]][2] = object.Cid
            tri_mesh_ind[tri_cnt[None]][3] = object.Anid
            tri_mesh_ind[tri_cnt[None]][4] = object.Bnid
            tri_mesh_ind[tri_cnt[None]][5] = object.Cnid
            tri_mesh_ind[tri_cnt[None]][6] = object.Auvid
            tri_mesh_ind[tri_cnt[None]][7] = object.Buvid
            tri_mesh_ind[tri_cnt[None]][8] = object.Cuvid
            tri_mesh_material[tri_cnt[None]] = object.material
            objs_ind[objs_cnt[None]] = tri_cnt[None]
            tri_cnt[None] += 1
        objs_type[objs_cnt[None]] = type
        self.objects.append(objs_cnt[None])
        objs_cnt[None] += 1
    def clear(self):
        self.objects.clear()

sphere_cnt = ti.field(ti.i32, shape=())
sphere_cnt[None] = 0
sphere_centers = ti.Vector.field(3, float)
sphere_radius = ti.field(float)
sphere_material = ti.field(ti.i32)
ti.root.dense(ti.i, M).place(sphere_centers, sphere_radius, sphere_material)

@ti.data_oriented
class sphere:
    def __init__(self, cen, r, m):
        self.center = cen
        self.radius = r
        self.material = m
    @ti.pyfunc
    def bounding_box(self, t0, t1):
        return True, AABB(self.center - ti.Vector([self.radius, self.radius, self.radius]), self.center + ti.Vector([self.radius, self.radius, self.radius]))
@ti.func
def get_sphere_uv(p):
    theta = ti.acos(-p[1])
    phi = ti.atan2(-p[2], p[0]) + Pi
    return phi / (2.0 * Pi), theta / Pi

@ti.func
def sphere_hit(id, r, t_min, t_max):
    oc = r.origin() - sphere_centers[id]
    a = r.direction().dot(r.direction())
    half_b = oc.dot(r.direction())
    c = oc.dot(oc) - sphere_radius[id] * sphere_radius[id]
    discriminant = half_b * half_b - a * c
    root = -1.0
    if (discriminant >= 0):
        sqrtd = ti.sqrt(discriminant)
        root = (-half_b - sqrtd) / a
        if (root < t_min or root > t_max):
            root = -1.0
    front_face, normal = False, ti.Vector([0.0, 0.0, 0.0])
    if (root >= 0.0):
        front_face, normal = set_face_normal(r, (r.at(root) - sphere_centers[id]) / sphere_radius[id])
    u, v = get_sphere_uv(normal)
    return root, r.at(root), front_face, normal, u, v, sphere_material[id]

moving_sphere_cnt = ti.field(ti.i32, shape=())
moving_sphere_cnt[None] = 0
moving_sphere_center0s = ti.Vector.field(3, float)
moving_sphere_center1s = ti.Vector.field(3, float)
moving_sphere_time0 = ti.field(float)
moving_sphere_time1 = ti.field(float)
moving_sphere_radius = ti.field(float)
moving_sphere_material = ti.field(ti.i32)
ti.root.dense(ti.i, M).place(moving_sphere_center0s, moving_sphere_center1s, moving_sphere_time0, moving_sphere_time1, moving_sphere_radius, moving_sphere_material)

@ti.data_oriented
class moving_sphere:
    def __init__(self, cen0, cen1, time0, time1, r, m):
        self.center0 = cen0
        self.center1 = cen1
        self.radius = r
        self.time0 = time0
        self.time1 = time1
        self.material = m
    @ti.pyfunc
    def bounding_box(self, t0, t1):
        return True, surrounding_box(AABB(self.center(t0) - ti.Vector([self.radius, self.radius, self.radius]), self.center + ti.Vector([self.radius, self.radius, self.radius])),
                                    AABB(self.center(t1) - ti.Vector([self.radius, self.radius, self.radius]), self.center + ti.Vector([self.radius, self.radius, self.radius])))
@ti.func
def moving_sphere_center(center0, center1, time0, time1, time):
    return center0 + ((time - time0) / (time1 - time0) * (center1 - center0))
@ti.func
def moving_sphere_hit(id, r, t_min, t_max):
    oc = r.origin() - moving_sphere_center(moving_sphere_center0s[id], moving_sphere_center1s[id], moving_sphere_time0[id], moving_sphere_time1[id], r.time())
    a = r.direction().dot(r.direction())
    half_b = oc.dot(r.direction())
    c = oc.dot(oc) - moving_sphere_radius[id] * moving_sphere_radius[id]
    discriminant = half_b * half_b - a * c
    root = -1.0
    if (discriminant >= 0):
        sqrtd = ti.sqrt(discriminant)
        root = (-half_b - sqrtd) / a
        if (root < t_min or root > t_max):
            root = -1.0
    front_face, normal = False, ti.Vector([0.0, 0.0, 0.0])
    if (root >= 0.0):
        front_face, normal = set_face_normal(r, (r.at(root) - moving_sphere_center(moving_sphere_center0s[id], moving_sphere_center1s[id], moving_sphere_time0[id], moving_sphere_time1[id], r.time())) / moving_sphere_radius[id])
    u, v = get_sphere_uv(normal)
    return root, r.at(root), front_face, normal, u, v, moving_sphere_material[id]

tri_cnt = ti.field(ti.i32, shape=())
tri_cnt[None] = 0
tri_pos_cnt = ti.field(ti.i32, shape=())
tri_pos_cnt[None] = 0
tri_norm_cnt = ti.field(ti.i32, shape=())
tri_norm_cnt[None] = 0
tri_uv_cnt = ti.field(ti.i32, shape=())
tri_uv_cnt[None] = 0
tri_mesh_ind = ti.Vector.field(9, ti.i32, shape=(M))
tri_mesh_material = ti.field(ti.i32, shape=(M))
tri_pos = ti.Vector.field(3, float, shape=(M))
tri_norm = ti.Vector.field(3, float, shape=(M))
tri_uv = ti.Vector.field(2, float, shape=(M))

@ti.data_oriented
class tri_id:
    def __init__(self, Aid, Bid, Cid, Anid, Bnid, Cnid, Auvid, Buvid, Cuvid, material):
        self.Aid, self.Bid, self.Cid = Aid, Bid, Cid
        self.Anid, self.Bnid, self.Cnid = Anid, Bnid, Cnid
        self.Auvid, self.Buvid, self.Cuvid = Auvid, Buvid, Cuvid
        self.material = material

def tri_bounding(id, t0, t1):
    tmp_min[None] = tri_pos[tri_mesh_ind[id][0]]
    tmp_max[None] = tri_pos[tri_mesh_ind[id][0]]
    for i, j in ti.ndrange(3, 3):
        if tmp_min[None][j] > tri_pos[tri_mesh_ind[id][i]][j]: tmp_min[None][j] = tri_pos[tri_mesh_ind[id][i]][j]
    for i, j in ti.ndrange(3, 3):
        if tmp_max[None][j] < tri_pos[tri_mesh_ind[id][i]][j]: tmp_max[None][j] = tri_pos[tri_mesh_ind[id][i]][j]
    for i in ti.static(range(3)):
        if (ti.abs(tmp_max[None][i] - tmp_min[None][i]) < 1e-4): tmp_max[None][i] = tmp_min[None][i] + 1e-4
    return True, AABB(tmp_min[None], tmp_max[None])

@ti.func
def tri_hit(id, r, t_min, t_max):
    v0v1 = tri_pos[tri_mesh_ind[id][1]] - tri_pos[tri_mesh_ind[id][0]]
    v0v2 = tri_pos[tri_mesh_ind[id][2]] - tri_pos[tri_mesh_ind[id][0]]
    is_hit = True
    pvec = r.direction().cross(v0v2)
    det = v0v1.dot(pvec)
    invDet = 1.0 / det
    tvec = r.origin() - tri_pos[tri_mesh_ind[id][0]]
    u = tvec.dot(pvec) * invDet
    if (u < 0 or u > 1): is_hit = False 
    qvec = tvec.cross(v0v1)
    v = r.direction().dot(qvec) * invDet
    if (v < 0 or u + v > 1): is_hit = False
    t = -1.0
    front_face, normal, uv = True, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0])
    if is_hit:
        if tri_mesh_ind[id][3] >= 0: 
            front_face, normal = set_face_normal(r, u * tri_norm[tri_mesh_ind[id][3]] + v * tri_norm[tri_mesh_ind[id][4]] + (1 - u - v) * tri_norm[tri_mesh_ind[id][5]])
            if front_face: t = v0v2.dot(qvec) * invDet
        else: t = v0v2.dot(qvec) * invDet
        if (t < t_min or t > t_max): t = -1.0
    if tri_mesh_ind[id][6] >= 0: uv = u * tri_uv[tri_mesh_ind[id][6]] + v * tri_uv[tri_mesh_ind[id][7]] + (1 - u - v) * tri_uv[tri_mesh_ind[id][8]]
    return t, r.at(t), front_face, normal, uv[0], uv[1], tri_mesh_material[id]

world = hittable_list()

@ti.func
def ray_color(r, world):
    cnt = 0
    flag = True
    res = ti.Vector([1.0, 1.0, 1.0])
    t, p, front_face, normal, u, v, m = bvh_hit(r, 0.001, float('inf'))
    scattered_o, scattered_d, scattered_tm = r.origin(), r.direction(), r.time()
    if (t > 0):
        while (t > 0 and cnt <= max_depth and flag):
            cnt += 1
            flag, scattered_o, scattered_d, scattered_tm = scatter(m, ray(scattered_o, scattered_d, scattered_tm), p, normal, front_face, u, v)
            res *= get_albedo(material_kind[m], material_extra[m], material_albedo[m], u, v, p)
            t, p, front_face, normal, u, v, m = bvh_hit(ray(scattered_o, scattered_d), 0.001, float('inf'))     
        if (cnt > max_depth):
            res = ti.Vector([0, 0, 0])
    if (material_kind[m] != 3):
        unit_direction = scattered_d.normalized()
        t = 0.5 * (unit_direction[1] + 1.0)
        res *= (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])
    return res

img_filename = []
img = ti.Vector.field(3, float, shape=(TEXTURE_MAXX, TEXTURE_MAXY))

def add_texture(filename):
    @ti.kernel
    def cpy_img(xx:ti.i32, yy:ti.i32):
        for i, j in ti.ndrange(xx, yy):
            textures[textures_cnt[None], i, j] = img[i, j]
    if img_filename.count(filename) > 0:
        return img_filename.index(filename)
    img_filename.append(filename)
    nimg = ti.imread(filename)
    img.from_numpy(nimg)
    textures_size[textures_cnt[None]][0], textures_size[textures_cnt[None]][1] = nimg.shape[0], nimg.shape[1]
    cpy_img(nimg.shape[0], nimg.shape[1])
    textures_cnt[None] += 1
    return textures_cnt[None] - 1

mtl_id = []
mtl_name = []
    
def load_mtl(filename):
    fn = ""
    tx_nm = ""
    first_mtl = True
    albedo = ti.Vector([0.0, 0.0, 0.0])
    for line in open(filename, "r"):
        if line.startswith('#'):continue
        values = line.split()
        if (not values): continue
        if (values[0] == 'newmtl'):
            if not first_mtl:
                mtl_name.append(fn)
                mtl_id.append(materials_add(6, albedo, add_texture(tx_nm)))
            else: first_mtl = False
            fn = values[1]
        elif values[0] == 'map_Kd':
            tx_nm = values[1]
        elif values[0] == 'Ka':
            Ka = list(map(float, values[1:]))
            albedo = ti.Vector([Ka[0], Ka[1], Ka[2]])
    
    mtl_name.append(fn)
    mtl_id.append(materials_add(6, albedo, add_texture(tx_nm)))

def load_obj(filename, d, tx, ty, tz):
    mtl = materials_add(0, ti.Vector([0.75, 0.75, 0.75]), 1.0)
    st_tri_cnt = tri_pos_cnt[None] - 1
    st_uv_cnt = tri_uv_cnt[None] - 1
    st_norm_cnt = tri_norm_cnt[None] - 1
    for line in open(filename, "r"):
        if (line.startswith('#')): continue
        values = line.split()
        if (not values): continue
        if (values[0] == 'v'):
            v = list(map(float, values[1:4]))
            tri_pos[tri_pos_cnt[None]] = ti.Vector([d * v[0] + tx, d * v[1] + ty, d * v[2] + tz])
            tri_pos_cnt[None] += 1
        elif (values[0] == 'vn'):
            v = list(map(float, values[1:4]))
            tri_norm[tri_norm_cnt[None]] = ti.Vector([v[0], v[1], v[2]])
            tri_norm_cnt[None] += 1
        elif (values[0] == 'vt'):
            v = list(map(float, values[1:3]))
            tri_uv[tri_uv_cnt[None]] = ti.Vector([v[0], v[1]])
            tri_uv_cnt[None] += 1
        elif values[0] in ('usemtl', 'usemat'):
            mtl = mtl_id[mtl_name.index(values[1])]
        elif (values[0] == 'mtllib'):
            load_mtl(values[1])
        elif (values[0] == 'f'):
            face = []
            uv = []
            norm = []
            for v in values[1:]:
                w = v.split('/')
                face.append(w[0])
                if (len(w) > 1): uv.append(w[1])
                if (len(w) > 2): norm.append(w[2])
            f = list(map(int, face))
            if (len(norm) == 0):
                if (len(uv) == 0):
                    world.add(3, tri_id(f[0] + st_tri_cnt, f[1] + st_tri_cnt, f[2] + st_tri_cnt, -1, -1, -1, -1, -1, -1, mtl))
                else:
                    uuvv = list(map(int, uv))
                    world.add(3, tri_id(f[0] + st_tri_cnt, f[1] + st_tri_cnt, f[2] + st_tri_cnt, -1, -1, -1, uuvv[0] + st_uv_cnt, uuvv[1] + st_uv_cnt, uuvv[2] + st_uv_cnt, mtl))
            else:
                nm = list(map(int, norm))
                uuvv = list(map(int, uv))
                world.add(3, tri_id(f[0] + st_tri_cnt, f[1] + st_tri_cnt, f[2] + st_tri_cnt, nm[0] + st_norm_cnt, nm[1] + st_norm_cnt, nm[2] + st_norm_cnt, uuvv[0] + st_uv_cnt, uuvv[1] + st_uv_cnt, uuvv[2] + st_uv_cnt, mtl))

def gen_objects():
    material_ground = materials_add(4, ti.Vector([0.2, 0.3, 0.1]), 1.0)
    material_left_wall = materials_add(0, ti.Vector([0.0, 0.6, 0.0]), 1.0)
    material_right_wall = materials_add(0, ti.Vector([0.6, 0.0, 0.0]), 1.0)
    material_center = materials_add(6, ti.Vector([1.0, 1.0, 1.0]), add_texture('earthmap.jpg'))
    material_left = materials_add(2, ti.Vector([1.0, 1.0, 1.0]), 1.5)
    material_right = materials_add(1, ti.Vector([0.6, 0.8, 0.8]), 0.2)
    material_light = materials_add(3, ti.Vector([10.0, 10.0, 10.0]), 1.0)
    world.add(1, sphere(ti.Vector([0, -0.2, -1.5]), 0.3, material_center))
    world.add(1, sphere(ti.Vector([0.7, 0.0, -0.5]), 0.5, material_left))
    world.add(1, sphere(ti.Vector([-0.8, 0.2, -1.0]), 0.7, material_right))

    tri_pos[0] = ti.Vector([0.5, 2.49, -1.0])
    tri_pos[1] = ti.Vector([0.5, 2.49, 0.0])
    tri_pos[2] = ti.Vector([-0.5, 2.49, -1.0])
    tri_pos[3] = ti.Vector([-0.5, 2.49, 0.0])
    tri_norm[0] = ti.Vector([0, -1.0, 0])
    tri_norm[1] = ti.Vector([0, -1.0, 0])
    tri_norm[2] = ti.Vector([0, -1.0, 0])
    tri_norm[3] = ti.Vector([0, -1.0, 0])
    world.add(3, tri_id(0, 1, 2, 0, 1, 2, 0, 1, 2, material_light))
    world.add(3, tri_id(1, 2, 3, 1, 2, 3, 1, 2, 3, material_light))

    tri_pos[4] = ti.Vector([1.5, 2.5, -2])
    tri_pos[5] = ti.Vector([1.5, 2.5, 1])
    tri_pos[6] = ti.Vector([-1.5, 2.5, -2])
    tri_pos[7] = ti.Vector([-1.5, 2.5, 1])
    tri_norm[4] = ti.Vector([0, -1.0, 0])
    tri_norm[5] = ti.Vector([0, -1.0, 0])
    tri_norm[6] = ti.Vector([0, -1.0, 0])
    tri_norm[7] = ti.Vector([0, -1.0, 0])
    world.add(3, tri_id(4, 5, 6, 4, 5, 6, 4, 5, 6, material_ground))
    world.add(3, tri_id(5, 6, 7, 5, 6, 7, 5, 6, 7, material_ground))

    tri_pos[8] = ti.Vector([1.5, -0.5, 1])
    tri_pos[9] = ti.Vector([1.5, 2.5, 1])
    tri_pos[10] = ti.Vector([-1.5, -0.5, 1])
    tri_pos[11] = ti.Vector([-1.5, 2.5, 1])
    tri_norm[8] = ti.Vector([0, 0, -1.0])
    tri_norm[9] = ti.Vector([0, 0, -1.0])
    tri_norm[10] = ti.Vector([0, 0, -1.0])
    tri_norm[11] = ti.Vector([0, 0, -1.0])
    world.add(3, tri_id(8, 9, 10, 8, 9, 10, 8, 9, 10, material_ground))
    world.add(3, tri_id(9, 10, 11, 9, 10, 11, 9, 10, 11, material_ground))

    tri_pos[12] = ti.Vector([1.5, -0.5, -2])
    tri_pos[13] = ti.Vector([1.5, -0.5, 1])
    tri_pos[14] = ti.Vector([-1.5, -0.5, -2])
    tri_pos[15] = ti.Vector([-1.5, -0.5, 1])
    tri_norm[12] = ti.Vector([0, 1.0, 0])
    tri_norm[13] = ti.Vector([0, 1.0, 0])
    tri_norm[14] = ti.Vector([0, 1.0, 0])
    tri_norm[15] = ti.Vector([0, 1.0, 0])
    world.add(3, tri_id(12, 13, 14, 12, 13, 14, 12, 13, 14, material_ground))
    world.add(3, tri_id(13, 14, 15, 13, 14, 15, 13, 14, 15, material_ground))

    tri_pos[16] = ti.Vector([1.5, 2.5, -2])
    tri_pos[17] = ti.Vector([1.5, 2.5, 1])
    tri_pos[18] = ti.Vector([1.5, -0.5, -2])
    tri_pos[19] = ti.Vector([1.5, -0.5, 1])
    tri_norm[16] = ti.Vector([-1.0, 0.0, 0])
    tri_norm[17] = ti.Vector([-1.0, 0.0, 0])
    tri_norm[18] = ti.Vector([-1.0, 0.0, 0])
    tri_norm[19] = ti.Vector([-1.0, 0.0, 0])
    world.add(3, tri_id(16, 17, 18, 16, 17, 18, 16, 17, 18, material_left_wall))
    world.add(3, tri_id(17, 18, 19, 17, 18, 19, 17, 18, 19, material_left_wall))

    tri_pos[20] = ti.Vector([-1.5, 2.5, -2])
    tri_pos[21] = ti.Vector([-1.5, 2.5, 1])
    tri_pos[22] = ti.Vector([-1.5, -0.5, -2])
    tri_pos[23] = ti.Vector([-1.5, -0.5, 1])
    tri_norm[20] = ti.Vector([1.0, 0.0, 0])
    tri_norm[21] = ti.Vector([1.0, 0.0, 0])
    tri_norm[22] = ti.Vector([1.0, 0.0, 0])
    tri_norm[23] = ti.Vector([1.0, 0.0, 0])
    world.add(3, tri_id(20, 21, 22, 20, 21, 22, 20, 21, 22, material_right_wall))
    world.add(3, tri_id(21, 22, 23, 21, 22, 23, 21, 22, 23, material_right_wall))
    tri_pos_cnt[None], tri_norm_cnt[None], tri_uv_cnt[None] = 24, 24, 24
    load_obj('assets/bunny.obj', 0.1, 0.5, -0.5, -1.5)
    print(tri_pos_cnt[None])
    print("Generated Objects")

@ti.kernel
def paint(cnt : int):
    for i, j in pixels:
        col = ti.Vector.zero(float, 3)
        for k in range(samples_per_pixel):
            (u, v) = ((i + ti.random()) / image_width, (j + ti.random()) / image_height)
            r = cam.get_ray(u, v)
            col += ray_color(r, world)
        col /= samples_per_pixel
        radience[i, j] += col
        pixels[i, j] = ti.sqrt(radience[i, j] / ti.cast(cnt, float))

gui = ti.GUI("Tiny Ray Tracer", res = (image_width, image_height))
pixels = ti.Vector.field(3, dtype = float, shape = (image_width, image_height))
radience = ti.Vector.field(3, dtype = float, shape = (image_width, image_height))
gen_perlin_noise()
gen_objects()
pur_objs_cnt[None] = objs_cnt[None]
bvh_tree.append(bvh_node(0.0, 1.0))
build_bvh_tree(0, world.objects)
bvh_init(0, -1)
print("Built BVH")
cam = camera()
cnt = 0
is_recording = False
result_dir = "./output"
video_manager = ti.VideoManager(output_dir = result_dir, framerate = 20, automatic_build = False)
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.LMB:
            x, y = gui.get_cursor_pos()
            lookfrom[None][0] = x * 4.0 - 2.0
            lookfrom[None][1] = y * 2.0 - 0.5
            print("Lookfrom change to ", lookfrom[None])
            look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
            dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == ti.GUI.LEFT:
            look_v_t = ti.Vector([look_v[0] * ti.cos(look_alpha) - look_v[2] * ti.sin(look_alpha), look_v[1], look_v[0] * ti.sin(look_alpha) + look_v[2] * ti.cos(look_alpha)])
            lookfrom[None][0], lookfrom[None][1], lookfrom[None][2] = look_v_t[0] + lookat[None][0], look_v_t[1] + lookat[None][1], look_v_t[2] + lookat[None][2]
            print("Lookfrom change to ", lookfrom[None])
            look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
            dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == ti.GUI.RIGHT:
            look_v_t = ti.Vector([look_v[0] * ti.cos(look_alpha) + look_v[2] * ti.sin(look_alpha), look_v[1], - look_v[0] * ti.sin(look_alpha) + look_v[2] * ti.cos(look_alpha)])
            lookfrom[None][0], lookfrom[None][1], lookfrom[None][2] = look_v_t[0] + lookat[None][0], look_v_t[1] + lookat[None][1], look_v_t[2] + lookat[None][2]
            print("Lookfrom change to ", lookfrom[None])
            look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
            dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == ti.GUI.UP:
            tmp_xz = ti.sqrt(look_v[0] * look_v[0] + look_v[2] * look_v[2])
            tmp_xz_t = ti.cos(look_alpha) * tmp_xz - ti.sin(look_alpha) * look_v[1]
            look_v_t = ti.Vector([look_v[0] * tmp_xz_t / tmp_xz, ti.sin(look_alpha) * tmp_xz + ti.cos(look_alpha) * look_v[1], look_v[2] * tmp_xz_t / tmp_xz])
            lookfrom[None][0], lookfrom[None][1], lookfrom[None][2] = look_v_t[0] + lookat[None][0], look_v_t[1] + lookat[None][1], look_v_t[2] + lookat[None][2]
            print("Lookfrom change to ", lookfrom[None])
            look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
            dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == ti.GUI.DOWN:
            tmp_xz = ti.sqrt(look_v[0] * look_v[0] + look_v[2] * look_v[2])
            tmp_xz_t = ti.cos(look_alpha) * tmp_xz + ti.sin(look_alpha) * look_v[1]
            look_v_t = ti.Vector([look_v[0] * tmp_xz_t / tmp_xz, -ti.sin(look_alpha) * tmp_xz + ti.cos(look_alpha) * look_v[1], look_v[2] * tmp_xz_t / tmp_xz])
            lookfrom[None][0], lookfrom[None][1], lookfrom[None][2] = look_v_t[0] + lookat[None][0], look_v_t[1] + lookat[None][1], look_v_t[2] + lookat[None][2]
            print("Lookfrom change to ", lookfrom[None])
            look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
            dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == 's':
            print("Screenshot, spp:", samples_per_pixel * cnt)
            gui.set_image(pixels)
            gui.show("cornellbox.jpg")
        elif gui.event.key == 'j':
            vfov[None] += 5.0
            print("vfov change to ", vfov[None])
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == 'k':
            vfov[None] -= 5.0
            print("vfov change to ", vfov[None])
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == 'r':
            if (is_recording):
                print("Stop Recording")
                video_manager.make_video(gif = True)
            else:
                print("Start recording")
            is_recording = not is_recording
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
    cnt += 1
    paint(cnt)
    gui.set_image(pixels)
    gui.show()
    if (is_recording):
        video_manager.write_frame(pixels)