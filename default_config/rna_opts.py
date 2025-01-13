import tempfile

rna_opts = {}
# Default directories
rna_opts["MGbind_pdb_dir"] = "data_preparation/data/MGbind/"
rna_opts["decoy_pdb_dir"] = "data_preparation/data/MGdecoy/"
rna_opts["decoy2_pdb_dir"] = "data_preparation/data/MGdecoy2/"


rna_opts["out_replace_pdb_dir"] = "data_prepartion/data_preparation/00-raw_pdbs/"
rna_opts["out_replace_decoypdb_dir"] = "data_prepartion/data_preparation/00-decoy_pdbs/"
rna_opts["tmp_dir"] = "data_preparation/data_result/pdbqt/"

rna_opts["decoy_rna_pdbqt"] = "data_preparation/pdbqt/decoy_tmp_rna_pdbqt/"
rna_opts["decoy2_rna_pdbqt"] = "data_preparation/pdbqt/decoy2_tmp_rna_pdbqt/"


rna_opts["temp_mid"] = "data_preparation/temp_mid/"

rna_opts["decoy_pdbqt_to_pdbs"] = "data_preparation/decoy_pdbqt_to_pdbs/"
rna_opts["decoy2_pdbqt_to_pdbs"] = "data_preparation/decoy2_pdbqt_to_pdbs/"

rna_opts["Ply"] = "data_preparation/Out_Ply/"

rna_opts["decoy2_Out_Ply"] = "data_preparation/data_result/out_ply/decoy2_Out_Ply/"
rna_opts["decoy_Out_Ply"] = "data_preparation/data_result/out_ply/decoy_Out_Ply/"
rna_opts["bind_Out_Ply"] = "data_preparation/data_result/out_ply/bind_Out_Ply/"

rna_opts["ply_file_template"] = rna_opts["ply_chain_dir"] + "/{}.ply"
rna_opts["in_ply_file"] = rna_opts["ply_chain_dir"] + "/{}.ply"

rna_opts["Decoy_lable_touch"] = "data_preparation/Decoy_lable_touch/"

# Surface features
rna_opts["compute_iface"] = True
# Mesh resolution. Everything gets very slow if it is lower than 1.0
rna_opts["mesh_res"] = 1.0
rna_opts["feature_interpolation"] = True


# Coords params
rna_opts["radius"] = 12.0

# Neural network patch application specific parameters.
rna_opts["site"] = {}
rna_opts["site"]["max_shape_size"] = 100
rna_opts["site"]["n_conv_layers"] = 3
rna_opts["site"]["max_distance"] =12.0 # Radius for the neural network.
rna_opts["site"][
    "rna_precomputation_dir"
] = "precomputation_12A/precomputation/"

rna_opts["site"][
    "bind_data"
] = "precomputation_12A/bind_data/"

rna_opts["site"][
    "decoy_data"
] = "precomputation_12A/decoy_data/"


rna_opts["site"][
    "decoy2_data"
] = "precomputation_12A/decoy2_data/"


rna_opts["site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
rna_opts["site"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
rna_opts["site"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"




rna_opts["site"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
rna_opts["site"]["feat_mask"] = [1.0] * 5


custom_params = {}
# predict rna_sit
custom_params["model_dir"] = "nn_models/all_feat_3l/model_data/"
custom_params["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
custom_params["n_conv_layers"] = 3
custom_params["out_pred_dir"] = "output/all_feat/pred_data/"
custom_params["out_surf_dir"] = "output/all_feat/pred_surfaces/"

