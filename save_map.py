import osmnx as ox

G = ox.graph_from_place('Московская область', network_type='drive')
G_speed = ox.speed.add_edge_speeds(G)
G_travel_time = ox.speed.add_edge_travel_times(G_speed)
ox.io.save_graphml(G_travel_time)
