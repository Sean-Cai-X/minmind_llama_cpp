cmake_minimum_required(VERSION 3.20)

project(minmind_llm_cpp
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(minmind_core
    src/checkpoint.cpp
    src/config.cpp
    src/dataset.cpp
    src/kg.cpp
    src/math.cpp
    src/model.cpp
    src/sampling.cpp
    src/trainer.cpp
    src/tokenizer.cpp
)

target_include_directories(minmind_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(minmind_core PUBLIC cxx_std_17)

add_executable(minmind_cli
    apps/minmind_cli.cpp
)

target_link_libraries(minmind_cli PRIVATE minmind_core)

enable_testing()

add_executable(minmind_tests
    tests/test_main.cpp
    tests/attention_tests.cpp
    tests/block_tests.cpp
    tests/causallm_tests.cpp
    tests/config_tests.cpp
    tests/dataset_tests.cpp
    tests/ffn_tests.cpp
    tests/golden_forward_tests.cpp
    tests/golden_fixture_tests.cpp
    tests/golden_loader.cpp
    tests/kg_tests.cpp
    tests/loss_tests.cpp
    tests/rope_tests.cpp
    tests/rmsnorm_tests.cpp
    tests/tokenizer_tests.cpp
    tests/model_forward_tests.cpp
    tests/generation_tests.cpp
)

target_link_libraries(minmind_tests PRIVATE minmind_core)

add_test(NAME minmind_tests COMMAND minmind_tests)
set_tests_properties(minmind_tests
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_train_smoke
    COMMAND minmind_cli
        train
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/tiny_lm.jsonl"
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/tiny_lm_head_cli.ctest.ckpt"
        --epochs 4
        --learning-rate 0.25
)

add_test(
    NAME minmind_cli_prepare_text_smoke
    COMMAND minmind_cli
        prepare-text
        --input "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        --output "${CMAKE_CURRENT_BINARY_DIR}/prepared_text_cli.ctest.jsonl"
        --seq-len 8
        --max-samples 2
)

add_test(
    NAME minmind_cli_inspect_text_byte_json_smoke
    COMMAND minmind_cli
        inspect-text
        --input "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        --tokenizer byte
        --vocab-size 16
        --json
)

add_test(
    NAME minmind_cli_prepare_byte_text_smoke
    COMMAND minmind_cli
        prepare-text
        --input "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        --output "${CMAKE_CURRENT_BINARY_DIR}/prepared_byte_text_cli.ctest.jsonl"
        --tokenizer byte
        --vocab-size 16
        --vocab-out "${CMAKE_CURRENT_BINARY_DIR}/prepared_byte_text_cli.ctest.vocab"
        --seq-len 8
        --max-samples 2
)

add_test(
    NAME minmind_cli_prepare_text_nested_output_smoke
    COMMAND minmind_cli
        prepare-text
        --input "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        --output "${CMAKE_CURRENT_BINARY_DIR}/nested/prepare/prepared_text_cli.ctest.jsonl"
        --seq-len 8
        --max-samples 2
)

add_test(
    NAME minmind_cli_train_nested_checkpoint_smoke
    COMMAND minmind_cli
        train
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/tiny_lm.jsonl"
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/nested/checkpoints/tiny_lm_head_cli.ctest.ckpt"
        --epochs 1
        --learning-rate 0.25
)

add_test(
    NAME minmind_cli_prepare_config_byte_smoke
    COMMAND minmind_cli
        prepare-text
        --config "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/prepare_config_byte.json"
)
set_tests_properties(minmind_cli_prepare_config_byte_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_prepare_config_byte_json_smoke
    COMMAND minmind_cli
        prepare-text
        --input "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        --output "${CMAKE_CURRENT_BINARY_DIR}/prepared_byte_config_json_cli.ctest.jsonl"
        --tokenizer byte
        --vocab-size 16
        --vocab-out "${CMAKE_CURRENT_BINARY_DIR}/prepared_byte_config_json_cli.ctest.vocab"
        --seq-len 8
        --max-samples 2
        --json
)

add_test(
    NAME minmind_cli_train_vocab6_smoke
    COMMAND minmind_cli
        train
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/tiny_lm_vocab6.jsonl"
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/tiny_lm_vocab6_cli.ctest.ckpt"
        --vocab-size 6
        --epochs 1
        --learning-rate 0.05
)

add_test(
    NAME minmind_cli_train_vocab6_json_smoke
    COMMAND minmind_cli
        train
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/tiny_lm_vocab6.jsonl"
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/tiny_lm_vocab6_json_cli.ctest.ckpt"
        --vocab-size 6
        --epochs 1
        --learning-rate 0.05
        --json
)

add_test(
    NAME minmind_cli_train_lm_profile_json_smoke
    COMMAND minmind_cli
        train
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name lm_smoke_cpu
        --json
)
set_tests_properties(minmind_cli_train_lm_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_validate_lm_profile_json_smoke
    COMMAND minmind_cli
        validate-lm-profile
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name lm_smoke_cpu
        --input "0,1"
        --max-new 2
        --json
)
set_tests_properties(minmind_cli_validate_lm_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS minmind_cli_train_lm_profile_json_smoke
)

add_test(
    NAME minmind_cli_inspect_checkpoint_vocab6_json_smoke
    COMMAND minmind_cli
        inspect-checkpoint
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/tiny_lm_vocab6_json_cli.ctest.ckpt"
        --vocab-size 6
        --json
)
set_tests_properties(minmind_cli_inspect_checkpoint_vocab6_json_smoke
    PROPERTIES
        DEPENDS minmind_cli_train_vocab6_json_smoke
)

add_test(
    NAME minmind_cli_generate_checkpoint_vocab6_json_smoke
    COMMAND minmind_cli
        generate-checkpoint
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/tiny_lm_vocab6_json_cli.ctest.ckpt"
        --input "0,1"
        --vocab-size 6
        --max-new 2
        --json
)
set_tests_properties(minmind_cli_generate_checkpoint_vocab6_json_smoke
    PROPERTIES
        DEPENDS minmind_cli_train_vocab6_json_smoke
)

add_test(
    NAME minmind_cli_inspect_vocab6_smoke
    COMMAND minmind_cli
        inspect-dataset
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/tiny_lm_vocab6.jsonl"
        --vocab-size 6
)

add_test(
    NAME minmind_cli_inspect_vocab6_json_smoke
    COMMAND minmind_cli
        inspect-dataset
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/tiny_lm_vocab6.jsonl"
        --vocab-size 6
        --json
)

add_test(
    NAME minmind_cli_train_config_vocab6_smoke
    COMMAND minmind_cli
        train
        --config "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/train_config_vocab6.json"
)
set_tests_properties(minmind_cli_train_config_vocab6_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_pipeline_config_byte_smoke
    COMMAND minmind_cli
        pipeline
        --config "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/pipeline_config_byte.json"
)
set_tests_properties(minmind_cli_pipeline_config_byte_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_pipeline_config_byte_json_smoke
    COMMAND minmind_cli
        pipeline
        --config "${CMAKE_CURRENT_SOURCE_DIR}/testdata/train/pipeline_config_byte_json.json"
        --json
)
set_tests_properties(minmind_cli_pipeline_config_byte_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_pipeline_profile_json_smoke
    COMMAND minmind_cli
        pipeline
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name lm_pipeline_smoke_cpu
        --json
)
set_tests_properties(minmind_cli_pipeline_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_inspect_pipeline_byte_smoke
    COMMAND minmind_cli
        inspect-dataset
        --data "${CMAKE_CURRENT_SOURCE_DIR}/build/pipeline_byte_cli.ctest.jsonl"
        --vocab-size 16
)
set_tests_properties(minmind_cli_inspect_pipeline_byte_smoke
    PROPERTIES
        DEPENDS minmind_cli_pipeline_config_byte_smoke
)

add_test(
    NAME minmind_cli_train_prepared_byte_smoke
    COMMAND minmind_cli
        train
        --data "${CMAKE_CURRENT_BINARY_DIR}/prepared_byte_text_cli.ctest.jsonl"
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/prepared_byte_text_cli.ctest.ckpt"
        --vocab-size 16
        --epochs 1
        --learning-rate 0.05
)
set_tests_properties(minmind_cli_train_prepared_byte_smoke
    PROPERTIES
        DEPENDS minmind_cli_prepare_byte_text_smoke
)

add_test(
    NAME minmind_cli_train_hf_vocab4_smoke
    COMMAND minmind_cli
        train
        --data "${CMAKE_CURRENT_SOURCE_DIR}/data/processed/tiny_shakespeare_vocab4_seq16_128.jsonl"
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/tiny_shakespeare_vocab4_cli.ctest.ckpt"
        --epochs 1
        --learning-rate 0.05
)

add_test(
    NAME minmind_cli_train_kg_transe_json_smoke
    COMMAND minmind_cli
        train-kg
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/kg/tiny_kg.tsv"
        --model transe
        --entities 4
        --relations 2
        --dim 3
        --epochs 2
        --learning-rate 0.01
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/kg_transe_cli.ctest.ckpt"
        --json
)

add_test(
    NAME minmind_cli_inspect_kg_checkpoint_transe_json_smoke
    COMMAND minmind_cli
        inspect-kg-checkpoint
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/kg_transe_cli.ctest.ckpt"
        --model transe
        --entities 4
        --relations 2
        --dim 3
        --json
)
set_tests_properties(minmind_cli_inspect_kg_checkpoint_transe_json_smoke
    PROPERTIES
        DEPENDS minmind_cli_train_kg_transe_json_smoke
)

add_test(
    NAME minmind_cli_train_kg_profile_json_smoke
    COMMAND minmind_cli
        train-kg
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name kg_smoke_cpu
        --model transe
        --json
)
set_tests_properties(minmind_cli_train_kg_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_kg_pipeline_profile_json_smoke
    COMMAND minmind_cli
        kg-pipeline
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name kg_smoke_cpu
        --model transe
        --json
)
set_tests_properties(minmind_cli_kg_pipeline_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_inspect_kg_dataset_json_smoke
    COMMAND minmind_cli
        inspect-kg-dataset
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/kg/tiny_kg.tsv"
        --entities 4
        --relations 2
        --json
)

add_test(
    NAME minmind_cli_prepare_kg_named_json_smoke
    COMMAND minmind_cli
        prepare-kg
        --input "${CMAKE_CURRENT_SOURCE_DIR}/testdata/kg/tiny_kg_named.tsv"
        --output "${CMAKE_CURRENT_BINARY_DIR}/tiny_kg_named_prepared.ctest.tsv"
        --entity-map "${CMAKE_CURRENT_BINARY_DIR}/tiny_kg_named_prepared.entities.ctest.tsv"
        --relation-map "${CMAKE_CURRENT_BINARY_DIR}/tiny_kg_named_prepared.relations.ctest.tsv"
        --json
)

add_test(
    NAME minmind_cli_inspect_prepared_kg_named_json_smoke
    COMMAND minmind_cli
        inspect-kg-dataset
        --data "${CMAKE_CURRENT_BINARY_DIR}/tiny_kg_named_prepared.ctest.tsv"
        --entities 4
        --relations 2
        --json
)
set_tests_properties(minmind_cli_inspect_prepared_kg_named_json_smoke
    PROPERTIES
        DEPENDS minmind_cli_prepare_kg_named_json_smoke
)

add_test(
    NAME minmind_cli_prepare_kg_pipeline_json_smoke
    COMMAND minmind_cli
        prepare-kg-pipeline
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name kg_prepare_pipeline_smoke_cpu
        --target kg
        --model transe
        --json
)
set_tests_properties(minmind_cli_prepare_kg_pipeline_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_prepare_kg_pipeline_rgcn_json_smoke
    COMMAND minmind_cli
        prepare-kg-pipeline
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name rgcn_prepare_pipeline_smoke_cpu
        --target rgcn
        --json
)
set_tests_properties(minmind_cli_prepare_kg_pipeline_rgcn_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_validate_kg_profile_json_smoke
    COMMAND minmind_cli
        validate-kg-profile
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name kg_smoke_cpu
        --model transe
        --json
)
set_tests_properties(minmind_cli_validate_kg_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS minmind_cli_kg_pipeline_profile_json_smoke
)

add_test(
    NAME minmind_cli_validate_prepared_kg_pipeline_json_smoke
    COMMAND minmind_cli
        validate-kg-profile
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name kg_prepare_pipeline_smoke_cpu
        --model transe
        --json
)
set_tests_properties(minmind_cli_validate_prepared_kg_pipeline_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS minmind_cli_prepare_kg_pipeline_json_smoke
)

add_test(
    NAME minmind_cli_train_rgcn_profile_json_smoke
    COMMAND minmind_cli
        train-rgcn
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name rgcn_smoke_cpu
        --json
)
set_tests_properties(minmind_cli_train_rgcn_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_inspect_rgcn_checkpoint_json_smoke
    COMMAND minmind_cli
        inspect-rgcn-checkpoint
        --checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/build/rgcn_smoke_cpu.ctest.ckpt"
        --nodes 4
        --relations 2
        --dim 3
        --json
)
set_tests_properties(minmind_cli_inspect_rgcn_checkpoint_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS minmind_cli_train_rgcn_profile_json_smoke
)

add_test(
    NAME minmind_cli_rgcn_pipeline_profile_json_smoke
    COMMAND minmind_cli
        rgcn-pipeline
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name rgcn_smoke_cpu
        --json
)
set_tests_properties(minmind_cli_rgcn_pipeline_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME minmind_cli_validate_rgcn_profile_json_smoke
    COMMAND minmind_cli
        validate-rgcn-profile
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name rgcn_smoke_cpu
        --json
)
set_tests_properties(minmind_cli_validate_rgcn_profile_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS minmind_cli_rgcn_pipeline_profile_json_smoke
)

add_test(
    NAME minmind_cli_validate_prepared_rgcn_pipeline_json_smoke
    COMMAND minmind_cli
        validate-rgcn-profile
        --profile "${CMAKE_CURRENT_SOURCE_DIR}/apps/minmind_training_profiles.json"
        --profile-name rgcn_prepare_pipeline_smoke_cpu
        --json
)
set_tests_properties(minmind_cli_validate_prepared_rgcn_pipeline_json_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS minmind_cli_prepare_kg_pipeline_rgcn_json_smoke
)

add_test(
    NAME minmind_cli_train_kg_distmult_json_smoke
    COMMAND minmind_cli
        train-kg
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/kg/tiny_kg.tsv"
        --model distmult
        --entities 4
        --relations 2
        --dim 3
        --epochs 2
        --learning-rate 0.01
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/kg_distmult_cli.ctest.ckpt"
        --json
)

add_test(
    NAME minmind_cli_train_kg_rotate_json_smoke
    COMMAND minmind_cli
        train-kg
        --data "${CMAKE_CURRENT_SOURCE_DIR}/testdata/kg/tiny_kg.tsv"
        --model rotate
        --entities 4
        --relations 2
        --dim 3
        --epochs 2
        --learning-rate 0.01
        --checkpoint "${CMAKE_CURRENT_BINARY_DIR}/kg_rotate_cli.ctest.ckpt"
        --json
)
