#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "kalki/common/config.h"
#include "kalki/common/thread_pool.h"
#include "kalki/common/types.h"
#include "kalki/metadata/metadata_store.h"
#include "kalki/query/similarity.h"

namespace kalki {

class QueryCoordinator {
 public:
  QueryCoordinator(const DatabaseConfig& config, MetadataStore* metadata_store,
                   ThreadPool* thread_pool);

  absl::StatusOr<QueryExecutionResult> Query(const std::string& query, const QueryFilter& filter);

 private:
  absl::StatusOr<std::vector<QueryRecord>> ProcessBlockGroup(
      const std::string& query, const QueryFilter& filter,
      const std::vector<BlockMetadata>& group) const;

  DatabaseConfig config_;
  MetadataStore* metadata_store_;
  ThreadPool* thread_pool_;
  SimilarityEngine similarity_engine_;
};

}  // namespace kalki
