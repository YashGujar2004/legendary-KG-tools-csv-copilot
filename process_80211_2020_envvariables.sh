
export SKIP_PREFIXES_FOR_FEATURES="9."   # skip anything starting with "9."
export SKIP_SECTIONS_FOR_FEATURES="9"    # skip exact section "9"

export OPENAI_KEY="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"
export SPEC_ANNEX_START_PAGE_NUMBER=3430
export CHUNK_ID_COUNTER="chunk-counter.txt"
export SPEC="80211.pdf"
export FULL_SPEC_WIFI="80211.pdf"
export SPEC_WIFI="80211-2020-sections.pdf"
export SPEC_CHUNKS="80211-2020-chunks.json"
export SPEC_CHUNKS_TRIM="80211-2020-chunks-trim.json"
export CHUNK_LINK_FRAME="80211-2020-chunks-link-frames.csv"
export CHUNK_LINK_FIELD="80211-2020-chunks-link-field.csv"
export CHUNK_LINK_ELEMENT="80211-2020-chunks-link-element.csv"
export FRAME_LINK_FIELD="80211-2020-frames-link-field.csv"
export FRAME_LINK_ELEMENT="80211-2020-frames-link-element.csv"

export FRAMES_LIST="frames_list.json"
export FRAMES_LIST_NAMES="frames_names_list.csv"

export FIELDS_LIST="fields_list.json"
export FIELDS_LIST_NAMES="fields_names_list.csv"

export ELEMENTS_LIST="elements_list.json"
export ELEMENTS_LIST_NAMES="elements_names_list.csv"

export FEATURE_SUBFEATURE_GROUP="features_subfeatures_groups.csv"
export TREE_TOPICS_FEATURES_SUBFEATURES="tree_topics_features_subfeatures.csv"
export KG_TOPICS_FEATURES_SUBFEATURES="kg_topics_features_subfeatures.csv"

export KG_TOPICS_FEATURE_NODES_FRAME_EDGES="kg_topics_feature_nodes_frame_edges.json"
export KG_TRANSFORM_FEATURE_NODES_FRAME_EDGES="kg_transformed_feature_nodes_frame_edges.json"

export KG_TRANSFORM_FEATURE_FOR_NEO4J="kg_transform_feature_neo4j.json"
export KG_BUILD_NEO4J_LEVEL="neo4j_input_level"
export CHUNKS_FOR_NEO4J="chunks_for_neo4j.csv"

export KG_TOP_FEATURES="kg_top_features_from_clusters.csv"
export KG_FEATURE_FRAME_SCORE="kg_feature_frame_scores.csv"

export KG_FEATURE_FIELD_SCORE="kg_feature_field_scores.csv"
export KG_FEATURE_ELEMENT_SCORE="kg_feature_element_scores.csv"

export KG_CLUSTERS="kg_clusters.json"
export KG_CLUSTER_PRIMARY_FRAME="kg_cluster_primary_frame.csv"
    
# --- Config ---
export EMBEDDING_MODEL="text-embedding-3-large"
export EMBEDDING_CACHE_FILE="embeddings_cache.json"
export SPEC_SECTIONS_TREE_HIERARCHY_FILE="tree_hierarchy.json"

export CHROMADB_FOR_CLUSTERS="vectorstore_clusters"
export CHROMADB_CLUSTER_COLLECTION_NAME="vectorstore_cluster_db" 

export CHROMADB_FOR_CLUSTER_SECTION="vectorstore_cluster_section"
export CHROMADB_CLUSTER_SECTION_COLLECTION_NAME="vectorstore_cluster_section_db" 

#------ DEFINES FOR VAR CREATION ---from TABLES-

export VAR_TABLE_LIST="var_table_list.json" 
export VAR_SECTION_LIST="var_section_list.json"
export VAR_ALL_SPEC="var_all_spec_vars.json"

export CHROMADB_CLOUD_SERVICE_MODE="true"
export CHROMADB_CLOUD_SERVICE_NAME="chromadb"
