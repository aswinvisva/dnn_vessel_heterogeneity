import matplotlib
import torch
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score
from flowsom import flowsom
import seaborn as sns

from cnn_dataloader import VesselDataset, preprocess_input
from feature_extraction.cnn_feature_gen import CNNFeatureGen
from feature_extraction.sift_feature_gen import SIFTFeatureGen
from models.clustering_helper import ClusteringHelper
from models.flowsom_clustering import ClusteringFlowSOM
from models.vessel_net import VesselNet
from models.s_lda import SpatialLDA
from utils.mibi_reader import get_all_point_data
from utils.extract_vessel_contours import *
from utils.markers_feature_gen import *
from utils.visualizer import vessel_nonvessel_heatmap, point_region_plots, vessel_region_plots, brain_region_plots, \
    all_points_plots, brain_region_expansion_heatmap, marker_expression_masks, vessel_areas_histogram
import config.config_settings as config


def get_mask_expression():
    marker_segmentation_masks, all_points_marker_data, marker_names = get_all_point_data()

    all_points_vessel_contours = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours, removed_contours = extract(segmentation_mask)
        all_points_vessel_contours.append(contours)

    # Store all data in lists
    n_points = config.n_points

    current_expansion_data = []

    # Iterate through each point
    for i in range(n_points):
        contours = all_points_vessel_contours[i]
        marker_data = all_points_marker_data[i]
        start_expression = datetime.datetime.now()

        # If we are on the first expansion, calculate the marker expression within the vessel itself. Otherwise,
        # calculate the marker expression in the outward microenvironment

        data = calculate_composition_marker_expression(marker_data, contours, marker_names,
                                                       point_num=i + 1)

        end_expression = datetime.datetime.now()

        print("Finished calculating expression for Point %s in %s" % (str(i + 1), end_expression - start_expression))

        current_expansion_data.append(data)

    all_expansions_features = pd.concat(current_expansion_data).fillna(0)

    idx = pd.IndexSlice
    all_expansions_features = all_expansions_features.loc[idx[:, :, :, "Data"], :]

    scaling_factor = config.scaling_factor
    transformation = config.transformation_type
    normalization = config.normalization_type
    n_markers = config.n_markers

    all_expansions_features = normalize_expression_data(all_expansions_features,
                                                        transformation=transformation,
                                                        normalization=normalization,
                                                        scaling_factor=scaling_factor,
                                                        n_markers=n_markers)

    all_expansions_features.index.rename(['Point', 'Vessel', 'Expansion', 'Data Type'], inplace=True)
    all_expansions_features = all_expansions_features.sort_index()

    all_expansions_features.to_csv("marker_mask_expression.csv")


def get_mask_microenvironment_expression():
    marker_segmentation_masks, all_points_marker_data, marker_names = get_all_point_data()
    pixel_interval = config.pixel_interval
    n_expansions = 2

    all_points_vessel_contours = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours, removed_contours = extract(segmentation_mask)
        all_points_vessel_contours.append(contours)

    # Store all data in lists
    expansion_data = []
    n_points = config.n_points

    # Iterate through each expansion
    for x in range(n_expansions):

        current_expansion_data = []

        all_points_stopped_vessels = 0

        # Iterate through each point
        for i in range(n_points):
            contours = all_points_vessel_contours[i]
            marker_data = all_points_marker_data[i]
            start_expression = datetime.datetime.now()

            # If we are on the first expansion, calculate the marker expression within the vessel itself. Otherwise,
            # calculate the marker expression in the outward microenvironment

            if x == 0:
                data = calculate_composition_marker_expression(marker_data, contours, marker_names,
                                                               point_num=i + 1)
            else:
                data, expression_images, stopped_vessels = calculate_microenvironment_marker_expression(
                    marker_data,
                    contours,
                    marker_names,
                    pixel_expansion_upper_bound=10,
                    pixel_expansion_lower_bound=0,
                    point_num=i + 1,
                    expansion_num=x)

                all_points_stopped_vessels += stopped_vessels

            end_expression = datetime.datetime.now()

            print(
                "Finished calculating expression for Point %s in %s" % (str(i + 1), end_expression - start_expression))

            current_expansion_data.append(data)

        print("There were %s vessels which could not expand inward/outward by %s pixels" % (
            all_points_stopped_vessels, x * pixel_interval))

        all_points_features = pd.concat(current_expansion_data).fillna(0)
        expansion_data.append(all_points_features)

    all_expansions_features = pd.concat(expansion_data).fillna(0)
    idx = pd.IndexSlice
    all_expansions_features = all_expansions_features.loc[idx[:, :, :, "Data"], :]

    scaling_factor = config.scaling_factor
    transformation = config.transformation_type
    normalization = config.normalization_type
    n_markers = config.n_markers

    all_expansions_features = normalize_expression_data(all_expansions_features,
                                                        transformation=transformation,
                                                        normalization=normalization,
                                                        scaling_factor=scaling_factor,
                                                        n_markers=n_markers)

    all_expansions_features.index.rename(['Point', 'Vessel', 'Expansion', 'Data Type'], inplace=True)
    all_expansions_features = all_expansions_features.sort_index()
    all_expansions_features = all_expansions_features.groupby(['Point', 'Vessel']).mean()

    all_expansions_features.to_csv("marker_expression_mask_and_expansion.csv")


def get_microenvironment_expression():
    marker_segmentation_masks, all_points_marker_data, markers_names = get_all_point_data()
    n_points = config.n_points
    pixel_interval = config.pixel_interval
    n_expansions = 2

    all_points_vessel_contours = []
    all_points_microenvironment_expression = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours, removed_contours = extract(segmentation_mask)
        all_points_vessel_contours.append(contours)

    # Iterate through each point
    for i in range(n_points):
        contours = all_points_vessel_contours[i]
        marker_data = all_points_marker_data[i]
        start_expression = datetime.datetime.now()

        # If we are on the first expansion, calculate the marker expression within the vessel itself. Otherwise,
        # calculate the marker expression in the outward microenvironment

        data, _, _ = calculate_microenvironment_marker_expression(
            marker_data,
            contours,
            markers_names,
            pixel_expansion_upper_bound=10,
            pixel_expansion_lower_bound=0,
            point_num=i + 1,
            expansion_num=1)

        end_expression = datetime.datetime.now()

        print("Finished calculating expression for Point %s in %s" % (str(i + 1), end_expression - start_expression))

        all_points_microenvironment_expression.append(data)

    all_samples_features = pd.concat(all_points_microenvironment_expression).fillna(0)

    idx = pd.IndexSlice
    all_samples_features = all_samples_features.loc[idx[:, :, :, "Data"], :]

    scaling_factor = config.scaling_factor
    transformation = config.transformation_type
    normalization = config.normalization_type
    n_markers = config.n_markers

    all_expansions_features = normalize_expression_data(all_samples_features,
                                                        transformation=transformation,
                                                        normalization=normalization,
                                                        scaling_factor=scaling_factor,
                                                        n_markers=n_markers)

    all_points_vessel_coords = []

    for i in range(n_points):
        contours = all_points_vessel_contours[i]
        coords = []

        for cnt in contours:
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            coords.append([cX, cY])

        all_points_vessel_coords.append(coords)

    all_expansions_features.to_csv("marker_expression_expansion_only.csv")


# def spatial_lda():
#     s_lda = SpatialLDA(n_topics=25, x_labels=markers_names, load=True)
#     d = s_lda.fit_predict(all_points_microenvironment_expression, all_points_vessel_coords)
#     s_lda.plot()

def find_vessel_clusters():
    marker_segmentation_masks, all_points_marker_data, markers_names = get_all_point_data()

    all_points_vessel_contours = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours, removed_contours = extract(segmentation_mask)
        all_points_vessel_contours.append(contours)

    # fsom = flowsom("marker_expression_mask_and_expansion.csv", if_fcs=False, if_drop=True, drop_col=['Unnamed: 0',
    #                                                                                                  'Unnamed: 1',
    #                                                                                                  'Unnamed: 2',
    #                                                                                                  'Unnamed: 3'])
    # read the data
    #

    fsom = flowsom("marker_mask_expression.csv", if_fcs=False, if_drop=True, drop_col=['Point',
                                                                                       'Vessel',
                                                                                       'Expansion',
                                                                                       'Data Type'])  # read the data

    fsom.som_mapping(30, 30, 34,
                     sigma=2.5,
                     lr=0.5,
                     batch_size=1000,
                     if_fcs=False)  # trains SOM with 100 iterations
    fsom.meta_clustering(KMeans, min_n=30,
                         max_n=55,
                         iter_n=3)  # train the meta clustering for cluster in range(40,45)

    fsom.labeling()

    output_dir = "%s/flowsom" % config.visualization_results_dir
    mkdir_p(output_dir)

    fsom.vis(t=4,  # the number of total nodes = t * bestk
             edge_color='b',
             node_size=300,
             with_labels=False)

    plt.savefig("%s/cluster_vis.png" % output_dir)
    plt.clf()

    output_tf_df = fsom.tf_df  # new column added: category

    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [[norm(-1.0), "midnightblue"],
              [norm(-0.5), "seagreen"],
              [norm(0.5), "mediumspringgreen"],
              [norm(1.0), "yellow"]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.get_cmap('tab20')(np.linspace(0, 20, 1))
    colors2 = plt.get_cmap('tab20b')(np.linspace(0, 20, 1))
    colors3 = plt.get_cmap('tab20c')(np.linspace(0, 20, 1))

    # combine them and build a new colormap
    colors_map_combined = np.vstack((colors1, colors2, colors3))
    mymap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors_map_combined)

    colors = [list(mymap(i / 20)) for i in range(20)]

    color_dict = {}

    for category in sorted(set(output_tf_df['category'].values)):
        color_dict[category] = colors[category]

    mmm = output_tf_df.groupby(['category']).mean()
    plt.figure(figsize=(20, 10))
    handlelist = [plt.plot([], marker="o", ls="", color=color_dict[key])[0] for key in color_dict.keys()]
    plt.legend(handlelist, sorted(set(output_tf_df['category'].values)), loc='upper right', bbox_to_anchor=(1.065, 1))

    x = [i for i in range(34)]
    my_xticks = mmm.columns
    plt.xticks(x, my_xticks)
    for i in range(len(mmm)):
        plt.plot(x, mmm.iloc[i], label=str(i))
    plt.xticks(rotation=60)
    plt.savefig("%s/cluster_line_plot.png" % output_dir)
    plt.clf()

    handlelist = [plt.plot([], marker="o", ls="", color=color_dict[key])[0] for key in color_dict.keys()]
    plt.legend(handlelist, sorted(set(output_tf_df['category'].values)), loc='upper right', bbox_to_anchor=(1.065, 1))

    ax = sns.heatmap(mmm, linewidth=0.5, xticklabels=my_xticks, cmap=cmap)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.savefig("%s/cluster_heatmap.png" % output_dir)
    plt.clf()

    cate_list = []
    count_list = []
    percent_list = []

    base = len(output_tf_df)
    for i in np.unique(output_tf_df.category):
        cate_list.append(i)
        c = len(output_tf_df[output_tf_df['category'] == i])
        p = c / base
        count_list.append(c)
        percent_list.append(p)

    count_df = pd.DataFrame({'category': cate_list, 'count': count_list, 'percentage': percent_list})
    count_df.to_csv("%s/cluster_counts.csv" % output_dir)

    d = output_tf_df['category'].values.tolist()

    map_dict = fsom.map_som.labels_map(fsom.tf_matrix, d)
    print(map_dict)

    bmu_counts = np.zeros((fsom.weights.shape[0], fsom.weights.shape[1]))

    for i in range(len(fsom.tf_matrix)):
        # print the milestone

        xx = fsom.tf_matrix[i, :]  # fetch the sample data
        winner = fsom.map_som.winner(xx)  # make prediction, prediction = the closest entry location in the SOM
        c = fsom.map_class[winner]  # from the location info get cluster info
        bmu_counts[winner] += 1

    print(bmu_counts.shape)
    print(Counter(bmu_counts.flatten()))

    vess_id_dir = "%s/flowsom/Vessel ID Clusters Masks" % config.visualization_results_dir
    mkdir_p(vess_id_dir)

    vess_marker_expression_dir = "%s/flowsom/Vessel ID Clusters Marker Expression" % config.visualization_results_dir
    mkdir_p(vess_marker_expression_dir)

    idx = 0
    for point_idx, per_point_contours in enumerate(all_points_vessel_contours):
        img = np.zeros((config.segmentation_mask_size[0], config.segmentation_mask_size[1], 3), np.uint8)

        marker_data = all_points_marker_data[point_idx]

        marker_dict = dict(zip(markers_names, marker_data))
        data = []

        for marker in config.marker_clusters["Vessels"]:
            data.append(marker_dict[marker])

        data = np.nanmean(np.array(data), axis=0)
        blurred_data = gaussian_filter(data, sigma=4)

        cm = plt.get_cmap('jet')
        colored_image = cm(blurred_data)

        for i in range(len(per_point_contours)):
            color = color_dict[d[idx]]
            cv.drawContours(colored_image, per_point_contours, i, color, 4)
            cv.drawContours(img, per_point_contours, i,
                            (int(color[0] * 255.0), int(color[1] * 255.0), int(color[2] * 255.0)), cv.FILLED)
            idx += 1

        plt.imshow(img)
        handlelist = [plt.plot([], marker="o", ls="", color=color_dict[key])[0] for key in color_dict.keys()]
        plt.legend(handlelist, sorted(set(output_tf_df['category'].values)), loc='upper left',
                   bbox_to_anchor=(1.05, 1))
        plt.savefig("%s/Point%s.png" % (vess_id_dir, str(point_idx + 1)))
        plt.clf()

        plt.imshow(colored_image)
        handlelist = [plt.plot([], marker="o", ls="", color=color_dict[key])[0] for key in color_dict.keys()]
        plt.legend(handlelist, sorted(set(output_tf_df['category'].values)), loc='upper left',
                   bbox_to_anchor=(1.05, 1))
        plt.savefig("%s/Point%s.png" % (vess_marker_expression_dir, str(point_idx + 1)))
        plt.clf()


def extract_vessel_heterogeneity(n=56,
                                 feature_extraction_method="vesselnet"):
    """
    Extract vessel heterogeneity
    :return:
    """

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data()

    contour_data_multiple_points = []
    contour_images_multiple_points = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours, removed_contours = extract(segmentation_mask, show=False)
        contour_data_multiple_points.append(contours)
        contour_images_multiple_points.append(contour_images)

    for i in range(len(contour_data_multiple_points)):
        contours = contour_data_multiple_points[i]
        marker_data = markers_data[i]
        get_microenvironment_masks(marker_data, contours, pixel_expansion_upper_bound=10, pixel_expansion_lower_bound=0)

    if feature_extraction_method == "vesselnet":
        data_loader = VesselDataset(contour_data_multiple_points, markers_data, batch_size=16, n=n)

        # vn = VesselNet(n=n)
        # vn.fit(data_loader, epochs=150)
        # vn.visualize_filters()
        # torch.save(vn.state_dict(), "trained_models/vessel_net_100.pth")

        vn = VesselNet(n=n)
        vn.load_state_dict(torch.load("trained_models/vessel_net_100.pth"))
        vn.to(torch.device("cuda"))
        # vn.visualize_filters()

        encoder_output = []
        marker_expressions = []

        for i in range(len(contour_data_multiple_points)):
            for x in range(len(contour_data_multiple_points[i])):
                contours = [contour_data_multiple_points[i][x]]
                marker_data = markers_data[i]
                expression, expression_images, stopped_vessels, _ = calculate_microenvironment_marker_expression(
                    marker_data,
                    contours,
                    plot=False)

                expression_image = preprocess_input(expression_images, n)
                expression_image = torch.unsqueeze(expression_image, 0)

                reconstructed_img, output = vn.forward(expression_image)

                y_pred_numpy = reconstructed_img.cpu().data.numpy()
                y_true_numpy = expression_image.cpu().data.numpy()

                row_i = np.random.choice(y_pred_numpy.shape[0], 1)
                random_pic = y_pred_numpy[row_i, :, :, :]
                random_pic = random_pic.reshape(34, n, n)

                true = y_true_numpy[row_i, :, :, :]
                true = true.reshape(34, n, n)

                # for w in range(len(random_pic)):
                #     cv.imshow("Predicted", random_pic[w] * 255)
                #     cv.imshow("True", true[w] * 255)
                #     cv.waitKey(0)

                output = output.cpu().data.numpy()
                encoder_output.append(output.reshape(2048))
                marker_expressions.append(expression[0])

    elif feature_extraction_method == "sift":
        flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

        sift = SIFTFeatureGen()
        encoder_output = sift.generate(flat_list)
    elif feature_extraction_method == "resnet":
        flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

        cnn = CNNFeatureGen()
        encoder_output = cnn.generate(flat_list)

    km = ClusteringHelper(encoder_output, n_clusters=10, metric="cosine", method="kmeans")
    indices, frequency = km.fit_predict()

    print(len(indices))

    indices = [[i, indices[i]] for i in range(len(indices))]
    values = set(map(lambda y: y[1], indices))

    grouped_indices = [[y[0] for y in indices if y[1] == x] for x in values]

    marker_expressions_grouped = []

    for i, cluster in enumerate(grouped_indices):
        temp = []
        for idx in cluster:
            temp.append(marker_expressions[idx])

        average = np.mean(temp, axis=0)

        marker_expressions_grouped.append(average)

    marker_expressions_grouped = np.array(marker_expressions_grouped)
    print(marker_expressions_grouped.shape)

    km.plot(x=marker_expressions_grouped, labels=markers_names)

    k = 10

    sampled_indices = []

    for cluster in grouped_indices:
        sampled_indices.append(random.choices(cluster, k=k))

    flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

    for i, cluster in enumerate(sampled_indices):
        for idx in cluster:
            cv.imshow("Cluster %s" % i, flat_list[idx])
            cv.waitKey(0)


if __name__ == '__main__':
    # get_mask_expression()
    # get_microenvironment_expression()
    find_vessel_clusters()
