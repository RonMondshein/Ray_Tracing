from surfaces.surface import Epsilon as Epsilon

def ray_intersections(ray, surfaces) -> list: 
    """""""""""""""""""""""""""""""""
    ray_intersections
    input - ray
    output - intersections
    find all the intersections of the ray with the surfaces
    """""""""""""""""""""""""""""""""
    rays_intersections = []
    for surface in surfaces:
        inter = surface.ray_intersection(ray)
        if inter is None:
            continue
        elif inter.distance < Epsilon:
            continue
        rays_intersections.append(inter)

    def get_distance(intersection):
        #for each intersection return the distance, for sorting
        return intersection.distance

    rays_intersections.sort(key=get_distance)
    return rays_intersections

