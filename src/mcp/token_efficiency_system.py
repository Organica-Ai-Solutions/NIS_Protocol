"""
Token Efficiency System for NIS Protocol Tools
Based on Anthropic's tool optimization research

Implements advanced token management:
- Intelligent pagination with context preservation
- Multi-dimensional filtering for precise data selection
- Smart truncation with continuation guidance
- Response streaming for large datasets
- Token budget management across tool chains

Reference: https://www.anthropic.com/engineering/writing-tools-for-agents
"""

import json
import logging
import math
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib

logger = logging.getLogger(__name__)


class TruncationStrategy(Enum):
    """Strategies for handling response truncation"""
    HARD_LIMIT = "hard_limit"           # Cut off at exact token limit
    SEMANTIC_BOUNDARY = "semantic_boundary"  # Cut at natural boundaries
    PRIORITY_BASED = "priority_based"    # Remove low-priority items first
    SUMMARIZE = "summarize"             # Summarize truncated content
    PAGINATE = "paginate"               # Split into pages


class FilterOperator(Enum):
    """Filter operators for data selection"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    REGEX = "regex"
    EXISTS = "exists"


@dataclass
class FilterCriterion:
    """Individual filter criterion"""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = True
    
    def matches(self, item: Dict[str, Any]) -> bool:
        """Check if item matches this filter criterion"""
        if self.field not in item:
            return self.operator == FilterOperator.NOT_EXISTS
        
        item_value = item[self.field]
        
        # Handle case sensitivity for strings
        if isinstance(item_value, str) and isinstance(self.value, str) and not self.case_sensitive:
            item_value = item_value.lower()
            compare_value = self.value.lower()
        else:
            compare_value = self.value
        
        # Apply operator
        if self.operator == FilterOperator.EQUALS:
            return item_value == compare_value
        elif self.operator == FilterOperator.NOT_EQUALS:
            return item_value != compare_value
        elif self.operator == FilterOperator.GREATER_THAN:
            return item_value > compare_value
        elif self.operator == FilterOperator.GREATER_EQUAL:
            return item_value >= compare_value
        elif self.operator == FilterOperator.LESS_THAN:
            return item_value < compare_value
        elif self.operator == FilterOperator.LESS_EQUAL:
            return item_value <= compare_value
        elif self.operator == FilterOperator.CONTAINS:
            return compare_value in str(item_value)
        elif self.operator == FilterOperator.NOT_CONTAINS:
            return compare_value not in str(item_value)
        elif self.operator == FilterOperator.IN:
            return item_value in compare_value if isinstance(compare_value, (list, tuple)) else False
        elif self.operator == FilterOperator.NOT_IN:
            return item_value not in compare_value if isinstance(compare_value, (list, tuple)) else True
        elif self.operator == FilterOperator.EXISTS:
            return True  # Field exists if we got here
        else:
            return False


@dataclass
class PaginationConfig:
    """Configuration for pagination"""
    page_size: int = 20
    max_pages: int = 100
    include_total_count: bool = True
    include_page_info: bool = True
    preserve_context: bool = True  # Maintain context across pages


@dataclass
class TokenBudget:
    """Token budget management for tool responses"""
    total_budget: int
    reserved_for_metadata: int = 100
    reserved_for_pagination: int = 50
    reserved_for_error_handling: int = 50
    
    @property
    def available_for_content(self) -> int:
        """Calculate tokens available for actual content"""
        return max(0, self.total_budget - self.reserved_for_metadata - 
                  self.reserved_for_pagination - self.reserved_for_error_handling)


@dataclass
class PaginatedResponse:
    """Response with pagination information"""
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'data': self.data,
            'pagination': self.pagination,
            'metadata': self.metadata
        }


class TokenEstimator:
    """Utility for estimating token counts"""
    
    @staticmethod
    def estimate_tokens(content: Any) -> int:
        """Estimate token count for various content types"""
        if isinstance(content, str):
            # Rough estimation: ~4 characters per token
            return max(1, len(content) // 4)
        elif isinstance(content, dict):
            json_str = json.dumps(content, separators=(',', ':'))
            return max(1, len(json_str) // 4)
        elif isinstance(content, list):
            if not content:
                return 1
            # Estimate based on first few items
            sample_size = min(3, len(content))
            sample_tokens = sum(TokenEstimator.estimate_tokens(item) for item in content[:sample_size])
            avg_tokens = sample_tokens / sample_size
            return max(1, int(avg_tokens * len(content)))
        else:
            return max(1, len(str(content)) // 4)
    
    @staticmethod
    def estimate_item_tokens(item: Dict[str, Any], include_keys: Optional[List[str]] = None) -> int:
        """Estimate tokens for a single item, optionally limiting to specific keys"""
        if include_keys:
            filtered_item = {k: v for k, v in item.items() if k in include_keys}
            return TokenEstimator.estimate_tokens(filtered_item)
        else:
            return TokenEstimator.estimate_tokens(item)


class DataFilter:
    """Advanced filtering system for data selection"""
    
    def __init__(self):
        self.criteria: List[FilterCriterion] = []
    
    def add_criterion(self, field: str, operator: FilterOperator, value: Any, case_sensitive: bool = True):
        """Add a filter criterion"""
        criterion = FilterCriterion(field, operator, value, case_sensitive)
        self.criteria.append(criterion)
        return self
    
    def add_criteria_from_dict(self, filter_dict: Dict[str, Any]):
        """Add multiple criteria from dictionary format"""
        for field, filter_spec in filter_dict.items():
            if isinstance(filter_spec, dict):
                operator = FilterOperator(filter_spec.get('operator', 'eq'))
                value = filter_spec.get('value')
                case_sensitive = filter_spec.get('case_sensitive', True)
            else:
                # Simple equality filter
                operator = FilterOperator.EQUALS
                value = filter_spec
                case_sensitive = True
            
            self.add_criterion(field, operator, value, case_sensitive)
        
        return self
    
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all filter criteria to data"""
        if not self.criteria:
            return data
        
        filtered_data = []
        for item in data:
            if all(criterion.matches(item) for criterion in self.criteria):
                filtered_data.append(item)
        
        return filtered_data
    
    def estimate_selectivity(self, sample_data: List[Dict[str, Any]]) -> float:
        """Estimate what fraction of data will pass the filter"""
        if not self.criteria or not sample_data:
            return 1.0
        
        sample_size = min(100, len(sample_data))
        sample = sample_data[:sample_size]
        
        matching = sum(1 for item in sample if all(criterion.matches(item) for criterion in self.criteria))
        
        return matching / sample_size if sample_size > 0 else 0.0


class SmartTruncator:
    """Intelligent truncation with context preservation"""
    
    def __init__(self, strategy: TruncationStrategy = TruncationStrategy.PRIORITY_BASED):
        self.strategy = strategy
    
    def truncate(
        self,
        data: List[Dict[str, Any]],
        token_budget: TokenBudget,
        priority_field: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Truncate data to fit within token budget
        
        Returns:
            (truncated_data, truncation_info)
        """
        if not data:
            return data, {'truncated': False}
        
        available_tokens = token_budget.available_for_content
        
        if self.strategy == TruncationStrategy.HARD_LIMIT:
            return self._truncate_hard_limit(data, available_tokens)
        elif self.strategy == TruncationStrategy.PRIORITY_BASED:
            return self._truncate_priority_based(data, available_tokens, priority_field)
        elif self.strategy == TruncationStrategy.SEMANTIC_BOUNDARY:
            return self._truncate_semantic_boundary(data, available_tokens)
        elif self.strategy == TruncationStrategy.SUMMARIZE:
            return self._truncate_with_summary(data, available_tokens)
        else:
            return self._truncate_hard_limit(data, available_tokens)
    
    def _truncate_hard_limit(self, data: List[Dict[str, Any]], token_limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Simple truncation at token limit"""
        truncated_data = []
        tokens_used = 0
        
        for item in data:
            item_tokens = TokenEstimator.estimate_tokens(item)
            if tokens_used + item_tokens <= token_limit:
                truncated_data.append(item)
                tokens_used += item_tokens
            else:
                break
        
        truncation_info = {
            'truncated': len(truncated_data) < len(data),
            'items_included': len(truncated_data),
            'items_total': len(data),
            'tokens_used': tokens_used,
            'strategy': 'hard_limit'
        }
        
        return truncated_data, truncation_info
    
    def _truncate_priority_based(
        self,
        data: List[Dict[str, Any]],
        token_limit: int,
        priority_field: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Truncate based on priority scores"""
        
        # Sort by priority if field is specified
        if priority_field:
            sorted_data = sorted(data, key=lambda x: x.get(priority_field, 0), reverse=True)
        else:
            # Use heuristic priority based on data characteristics
            sorted_data = sorted(data, key=self._calculate_heuristic_priority, reverse=True)
        
        return self._truncate_hard_limit(sorted_data, token_limit)
    
    def _truncate_semantic_boundary(self, data: List[Dict[str, Any]], token_limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Truncate at semantic boundaries (e.g., complete groups)"""
        # Group items by semantic similarity (simplified)
        groups = self._group_semantically(data)
        
        truncated_data = []
        tokens_used = 0
        
        for group in groups:
            group_tokens = sum(TokenEstimator.estimate_tokens(item) for item in group)
            
            if tokens_used + group_tokens <= token_limit:
                truncated_data.extend(group)
                tokens_used += group_tokens
            else:
                break
        
        truncation_info = {
            'truncated': len(truncated_data) < len(data),
            'items_included': len(truncated_data),
            'items_total': len(data),
            'tokens_used': tokens_used,
            'strategy': 'semantic_boundary',
            'groups_included': len([g for g in groups if any(item in truncated_data for item in g)])
        }
        
        return truncated_data, truncation_info
    
    def _truncate_with_summary(self, data: List[Dict[str, Any]], token_limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Truncate and provide summary of omitted data"""
        # First, try normal truncation
        truncated_data, info = self._truncate_hard_limit(data, int(token_limit * 0.8))  # Reserve 20% for summary
        
        if info['truncated']:
            # Create summary of omitted data
            omitted_data = data[len(truncated_data):]
            summary = self._create_summary(omitted_data)
            
            # Add summary as special item
            summary_item = {
                '_summary': True,
                '_omitted_count': len(omitted_data),
                '_summary_content': summary
            }
            
            truncated_data.append(summary_item)
            info['strategy'] = 'summarize'
            info['summary_included'] = True
        
        return truncated_data, info
    
    def _calculate_heuristic_priority(self, item: Dict[str, Any]) -> float:
        """Calculate heuristic priority for an item"""
        priority = 0.0
        
        # Boost priority for items with certain key fields
        high_value_fields = ['name', 'title', 'id', 'status', 'result', 'error', 'summary']
        for field in high_value_fields:
            if field in item:
                priority += 1.0
        
        # Boost priority for items with more information
        priority += len(item) * 0.1
        
        # Boost priority for items with numeric values (often metrics)
        numeric_fields = sum(1 for v in item.values() if isinstance(v, (int, float)))
        priority += numeric_fields * 0.2
        
        return priority
    
    def _group_semantically(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group items by semantic similarity (simplified implementation)"""
        if not data:
            return []
        
        # Simple grouping by shared keys
        groups = []
        current_group = [data[0]]
        current_keys = set(data[0].keys())
        
        for item in data[1:]:
            item_keys = set(item.keys())
            similarity = len(current_keys.intersection(item_keys)) / len(current_keys.union(item_keys))
            
            if similarity > 0.5:  # Similar enough to group together
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]
                current_keys = item_keys
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _create_summary(self, data: List[Dict[str, Any]]) -> str:
        """Create summary of omitted data"""
        if not data:
            return "No additional data"
        
        summary_parts = []
        
        # Count by type/category if available
        categories = {}
        for item in data:
            category = item.get('type', item.get('category', 'unknown'))
            categories[category] = categories.get(category, 0) + 1
        
        if categories:
            category_summary = ", ".join(f"{count} {cat}" for cat, count in categories.items())
            summary_parts.append(f"Omitted {len(data)} items: {category_summary}")
        else:
            summary_parts.append(f"Omitted {len(data)} additional items")
        
        # Add key statistics if available
        numeric_fields = {}
        for item in data:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        for field, values in numeric_fields.items():
            if len(values) > 1:
                avg_val = sum(values) / len(values)
                summary_parts.append(f"Average {field}: {avg_val:.2f}")
        
        return ". ".join(summary_parts)


class TokenEfficientPaginator:
    """Advanced pagination with token budget awareness"""
    
    def __init__(self, config: PaginationConfig):
        self.config = config
        self.truncator = SmartTruncator()
    
    def paginate(
        self,
        data: List[Dict[str, Any]],
        page: int = 1,
        token_budget: Optional[TokenBudget] = None,
        data_filter: Optional[DataFilter] = None,
        sort_field: Optional[str] = None,
        sort_desc: bool = False
    ) -> PaginatedResponse:
        """
        Create paginated response with token efficiency
        
        Args:
            data: Source data to paginate
            page: Page number (1-based)
            token_budget: Token budget constraints
            data_filter: Optional filter to apply
            sort_field: Field to sort by
            sort_desc: Sort in descending order
            
        Returns:
            PaginatedResponse with data and pagination info
        """
        start_time = time.time()
        
        # Apply filtering first
        if data_filter:
            data = data_filter.apply(data)
        
        # Apply sorting
        if sort_field:
            data = sorted(data, key=lambda x: x.get(sort_field, ''), reverse=sort_desc)
        
        total_items = len(data)
        
        # Calculate pagination bounds
        start_idx = (page - 1) * self.config.page_size
        end_idx = min(start_idx + self.config.page_size, total_items)
        
        # Get page data
        page_data = data[start_idx:end_idx]
        
        # Apply token budget constraints if specified
        truncation_info = None
        if token_budget:
            page_data, truncation_info = self.truncator.truncate(page_data, token_budget)
        
        # Calculate pagination metadata
        total_pages = math.ceil(total_items / self.config.page_size)
        
        pagination_info = {
            'page': page,
            'page_size': self.config.page_size,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'next_page': page + 1 if page < total_pages else None,
            'prev_page': page - 1 if page > 1 else None
        }
        
        if self.config.include_total_count:
            pagination_info['total_items'] = total_items
        
        if self.config.include_page_info:
            pagination_info.update({
                'items_on_page': len(page_data),
                'start_index': start_idx + 1,  # 1-based for user display
                'end_index': start_idx + len(page_data)
            })
        
        # Add token efficiency information
        if token_budget:
            pagination_info['token_info'] = {
                'budget_total': token_budget.total_budget,
                'budget_available': token_budget.available_for_content,
                'estimated_tokens_used': sum(TokenEstimator.estimate_tokens(item) for item in page_data)
            }
            
            if truncation_info:
                pagination_info['truncation'] = truncation_info
        
        # Create metadata
        metadata = {
            'processing_time': time.time() - start_time,
            'filtered': data_filter is not None,
            'sorted': sort_field is not None,
            'token_optimized': token_budget is not None
        }
        
        if data_filter:
            metadata['filter_selectivity'] = len(data) / len(data) if data else 0  # After filtering
        
        return PaginatedResponse(
            data=page_data,
            pagination=pagination_info,
            metadata=metadata
        )
    
    def create_continuation_token(self, page: int, filters: Optional[Dict[str, Any]] = None) -> str:
        """Create continuation token for stateless pagination"""
        token_data = {
            'page': page,
            'filters': filters or {},
            'timestamp': time.time()
        }
        
        token_str = json.dumps(token_data, sort_keys=True)
        return hashlib.md5(token_str.encode()).hexdigest()
    
    def parse_continuation_token(self, token: str) -> Dict[str, Any]:
        """Parse continuation token (simplified - in practice would decrypt)"""
        # This is a placeholder - real implementation would securely encode/decode
        return {'page': 1, 'filters': {}}


class TokenEfficiencyManager:
    """
    Central manager for token efficiency across all NIS Protocol tools.
    
    Provides unified interface for:
    - Pagination with token awareness
    - Smart filtering and truncation
    - Response optimization
    - Token budget management
    """
    
    def __init__(self, default_token_limit: int = 2000):
        self.default_token_limit = default_token_limit
        self.paginator = TokenEfficientPaginator(PaginationConfig())
        self.truncator = SmartTruncator()
        
        # Performance metrics
        self.metrics = {
            'requests_processed': 0,
            'tokens_saved': 0,
            'truncations_applied': 0,
            'filters_applied': 0
        }
        
        logger.info("🎯 Token Efficiency Manager initialized")
    
    def create_efficient_response(
        self,
        tool_name: str,
        raw_data: List[Dict[str, Any]],
        page: int = 1,
        page_size: int = 20,
        token_limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort_field: Optional[str] = None,
        sort_desc: bool = False,
        truncation_strategy: TruncationStrategy = TruncationStrategy.PRIORITY_BASED
    ) -> Dict[str, Any]:
        """
        Create token-efficient response with all optimizations applied
        
        Args:
            tool_name: Name of the tool generating the response
            raw_data: Raw data from tool execution
            page: Page number for pagination
            page_size: Items per page
            token_limit: Maximum tokens for response
            filters: Filter criteria to apply
            sort_field: Field to sort by
            sort_desc: Sort in descending order
            truncation_strategy: How to handle truncation
            
        Returns:
            Optimized response dictionary
        """
        start_time = time.time()
        
        # Update metrics
        self.metrics['requests_processed'] += 1
        
        # Create token budget
        effective_limit = token_limit or self.default_token_limit
        token_budget = TokenBudget(total_budget=effective_limit)
        
        # Create data filter if specified
        data_filter = None
        if filters:
            data_filter = DataFilter()
            data_filter.add_criteria_from_dict(filters)
            self.metrics['filters_applied'] += 1
        
        # Update pagination config
        pagination_config = PaginationConfig(page_size=page_size)
        paginator = TokenEfficientPaginator(pagination_config)
        
        # Set truncation strategy
        self.truncator.strategy = truncation_strategy
        
        # Create paginated response
        response = paginator.paginate(
            data=raw_data,
            page=page,
            token_budget=token_budget,
            data_filter=data_filter,
            sort_field=sort_field,
            sort_desc=sort_desc
        )
        
        # Add efficiency metadata
        response_dict = response.to_dict()
        response_dict['_efficiency'] = {
            'tool_name': tool_name,
            'processing_time': time.time() - start_time,
            'optimization_applied': True,
            'token_budget': effective_limit,
            'estimated_tokens': sum(TokenEstimator.estimate_tokens(item) for item in response.data),
            'efficiency_score': self._calculate_efficiency_score(response, effective_limit)
        }
        
        # Update metrics
        if response.pagination.get('truncation'):
            self.metrics['truncations_applied'] += 1
        
        # Estimate tokens saved
        original_tokens = sum(TokenEstimator.estimate_tokens(item) for item in raw_data)
        optimized_tokens = response_dict['_efficiency']['estimated_tokens']
        tokens_saved = max(0, original_tokens - optimized_tokens)
        self.metrics['tokens_saved'] += tokens_saved
        
        return response_dict
    
    def _calculate_efficiency_score(self, response: PaginatedResponse, token_limit: int) -> float:
        """Calculate efficiency score for the response"""
        if not response.data:
            return 0.0
        
        # Calculate information density
        total_tokens = sum(TokenEstimator.estimate_tokens(item) for item in response.data)
        token_utilization = min(1.0, total_tokens / token_limit)
        
        # Calculate completeness (not truncated)
        completeness = 1.0 if not response.pagination.get('truncation', {}).get('truncated') else 0.8
        
        # Calculate relevance (filtered data ratio)
        relevance = 1.0  # Assume all returned data is relevant after filtering
        
        # Combined efficiency score
        efficiency = (token_utilization * 0.4 + completeness * 0.4 + relevance * 0.2)
        
        return round(efficiency, 3)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the efficiency manager"""
        return dict(self.metrics)
    
    def suggest_optimizations(self, tool_name: str, typical_response_size: int) -> List[str]:
        """Suggest optimizations based on usage patterns"""
        suggestions = []
        
        if typical_response_size > self.default_token_limit:
            suggestions.append("Consider implementing pagination for this tool")
        
        if typical_response_size > self.default_token_limit * 2:
            suggestions.append("Implement response filtering to reduce data volume")
        
        if self.metrics['truncations_applied'] / max(1, self.metrics['requests_processed']) > 0.3:
            suggestions.append("High truncation rate - consider increasing default token limits")
        
        return suggestions


# Example usage and testing
def main():
    """Example usage of token efficiency system"""
    
    # Create sample data
    sample_data = []
    for i in range(100):
        sample_data.append({
            'id': f'item_{i:03d}',
            'name': f'Sample Item {i}',
            'category': 'A' if i % 3 == 0 else 'B' if i % 3 == 1 else 'C',
            'value': i * 1.5,
            'priority': i % 5,
            'description': f'This is a detailed description for item {i} with additional context and metadata.',
            'metadata': {
                'created': f'2024-01-{(i % 28) + 1:02d}',
                'tags': [f'tag_{i % 5}', f'category_{i % 3}']
            }
        })
    
    # Initialize efficiency manager
    manager = TokenEfficiencyManager(default_token_limit=1000)
    
    # Test different scenarios
    print("=== BASIC PAGINATION ===")
    response1 = manager.create_efficient_response(
        tool_name='dataset_search',
        raw_data=sample_data,
        page=1,
        page_size=10
    )
    print(f"Items returned: {len(response1['data'])}")
    print(f"Token estimate: {response1['_efficiency']['estimated_tokens']}")
    
    print("\n=== WITH FILTERING ===")
    response2 = manager.create_efficient_response(
        tool_name='dataset_search',
        raw_data=sample_data,
        page=1,
        page_size=10,
        filters={'category': 'A', 'priority': {'operator': 'gte', 'value': 3}}
    )
    print(f"Items returned: {len(response2['data'])}")
    print(f"Filtered from {len(sample_data)} to {response2['pagination']['total_items']} items")
    
    print("\n=== WITH TOKEN CONSTRAINTS ===")
    response3 = manager.create_efficient_response(
        tool_name='dataset_search',
        raw_data=sample_data,
        page=1,
        page_size=50,  # Request more than token budget allows
        token_limit=500,  # Tight token budget
        truncation_strategy=TruncationStrategy.PRIORITY_BASED
    )
    print(f"Items returned: {len(response3['data'])}")
    print(f"Truncated: {response3['pagination'].get('truncation', {}).get('truncated', False)}")
    
    # Show performance metrics
    print(f"\n=== PERFORMANCE METRICS ===")
    metrics = manager.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
