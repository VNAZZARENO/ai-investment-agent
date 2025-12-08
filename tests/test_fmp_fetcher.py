"""
Unit tests for FMP (Financial Modeling Prep) API fetcher.

Tests basic functionality:
- API availability checking
- Request construction
- Response parsing
- Error handling (invalid key, rate limits, network errors)
- Data extraction from multiple endpoints
"""

import pytest
import aiohttp
import os
from unittest.mock import AsyncMock, patch, MagicMock
from src.data.fmp_fetcher import FMPFetcher, get_fmp_fetcher


class TestFMPFetcherInit:
    """Test FMPFetcher initialization."""
    
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        fetcher = FMPFetcher(api_key="test-key")
        
        assert fetcher.api_key == "test-key"
        assert fetcher.base_url == "https://financialmodelingprep.com/stable"
        assert fetcher._session is None
        assert fetcher._key_validated is False
    
    @patch.dict(os.environ, {'FMP_API_KEY': 'env-key'})
    def test_init_from_environment(self):
        """Test initialization from environment variable."""
        fetcher = FMPFetcher()
        
        assert fetcher.api_key == "env-key"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_api_key(self):
        """Test initialization without API key."""
        fetcher = FMPFetcher()
        
        assert fetcher.api_key is None
    
    def test_is_available_with_key(self):
        """Test that is_available returns True when key is present."""
        fetcher = FMPFetcher(api_key="test-key")
        
        assert fetcher.is_available() is True
    
    @patch.dict(os.environ, {}, clear=True)  # Clear FMP_API_KEY from env
    def test_is_available_without_key(self):
        """Test that is_available returns False when key is missing."""
        fetcher = FMPFetcher(api_key=None)
        
        assert fetcher.is_available() is False


class TestFMPFetcherContextManager:
    """Test async context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_context_manager_creates_session(self):
        """Test that entering context creates a session."""
        fetcher = FMPFetcher(api_key="test-key")
        
        async with fetcher:
            assert fetcher._session is not None
            assert isinstance(fetcher._session, aiohttp.ClientSession)
    
    @pytest.mark.asyncio
    async def test_context_manager_closes_session(self):
        """Test that exiting context closes the session."""
        fetcher = FMPFetcher(api_key="test-key")
        
        async with fetcher:
            session = fetcher._session
        
        # Session should be closed after exiting context
        assert session.closed


class TestFMPGet:
    """Test the internal _get method."""
    
    @pytest.mark.asyncio
    async def test_get_success(self):
        """Test successful API request."""
        fetcher = FMPFetcher(api_key="test-key")
        
        mock_data = [{'pe': 15.5, 'pb': 2.0}]
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_data)
        
        # Create proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        
        fetcher._session = mock_session
        
        result = await fetcher._get('ratios', {'symbol': 'AAPL', 'limit': 1})
        
        assert result == mock_data
        assert fetcher._key_validated is True
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_get_without_api_key(self):
        """Test that _get returns None when API key is not available."""
        # Clear environment so fetcher has no API key
        fetcher = FMPFetcher(api_key=None)
        
        # Early return when no API key - never hits the network
        result = await fetcher._get('ratios', {'symbol': 'AAPL'})
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_403_invalid_key(self):
        """Test handling of 403 with invalid API key."""
        fetcher = FMPFetcher(api_key="invalid-key")
        
        mock_response = MagicMock()
        mock_response.status = 403
        
        # Create proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        
        fetcher._session = mock_session
        
        with pytest.raises(ValueError, match="FMP_API_KEY is invalid"):
            await fetcher._get('ratios', {'symbol': 'AAPL'})
    
    @pytest.mark.asyncio
    async def test_get_403_rate_limit_after_validation(self):
        """Test handling of 403 after key was previously validated."""
        fetcher = FMPFetcher(api_key="test-key")
        fetcher._key_validated = True  # Key was previously validated
        
        mock_response = MagicMock()
        mock_response.status = 403
        
        # Create proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        
        fetcher._session = mock_session
        
        # Should return None instead of raising
        result = await fetcher._get('ratios', {'symbol': 'AAPL'})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_404_not_found(self):
        """Test handling of 404 (data not found)."""
        fetcher = FMPFetcher(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status = 404
        
        # Create proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        
        fetcher._session = mock_session
        
        result = await fetcher._get('ratios', {'symbol': 'INVALID'})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_network_error(self):
        """Test handling of network errors."""
        fetcher = FMPFetcher(api_key="test-key")
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Network failed"))
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        
        fetcher._session = mock_session
        
        result = await fetcher._get('ratios', {'symbol': 'AAPL'})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_adds_api_key_to_params(self):
        """Test that API key is added to request parameters."""
        fetcher = FMPFetcher(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])
        
        # Create proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        
        fetcher._session = mock_session
        
        await fetcher._get('ratios', {'symbol': 'AAPL'})
        
        # Verify API key was added to params
        call_args = mock_session.get.call_args
        params = call_args[1]['params']
        assert params['apikey'] == "test-key"


class TestGetFinancialMetrics:
    """Test comprehensive financial metrics fetching."""
    
    @pytest.mark.asyncio
    async def test_get_financial_metrics_all_sources(self):
        """Test fetching from all three endpoints."""
        fetcher = FMPFetcher(api_key="test-key")
        
        # Mock ratios endpoint
        ratios_data = [{
            'priceToEarningsRatio': 25.5,
            'priceToBookRatio': 10.2,
            'priceToEarningsGrowthRatio': 1.5,
            'currentRatio': 1.2,
            'debtToEquityRatio': 1.8,
            'netProfitMargin': 0.25
        }]
        
        # Mock key-metrics endpoint
        metrics_data = [{
            'returnOnEquity': 0.45,
            'returnOnAssets': 0.18
        }]
        
        # Mock growth endpoint
        growth_data = [{
            'growthRevenue': 0.15,
            'growthEPS': 0.20
        }]
        
        # Create a mock that returns different data based on endpoint
        async def mock_get(endpoint, params):
            if endpoint == 'ratios':
                return ratios_data
            elif endpoint == 'key-metrics':
                return metrics_data
            elif endpoint == 'income-statement-growth':
                return growth_data
            return None
        
        fetcher._get = AsyncMock(side_effect=mock_get)
        
        result = await fetcher.get_financial_metrics('AAPL')
        
        # Verify all fields are populated
        assert result['pe'] == 25.5
        assert result['pb'] == 10.2
        assert result['peg'] == 1.5
        assert result['roe'] == 0.45
        assert result['roa'] == 0.18
        assert result['current_ratio'] == 1.2
        assert result['debt_to_equity'] == 1.8
        assert result['profit_margin'] == 0.25
        assert result['revenue_growth'] == 0.15
        assert result['eps_growth'] == 0.20
        assert result['_source'] == 'fmp'
    
    @pytest.mark.asyncio
    async def test_get_financial_metrics_partial_data(self):
        """Test handling of partial data from endpoints."""
        fetcher = FMPFetcher(api_key="test-key")
        
        # Only ratios endpoint returns data
        async def mock_get(endpoint, params):
            if endpoint == 'ratios':
                return [{'priceToEarningsRatio': 15.0}]
            return None
        
        fetcher._get = AsyncMock(side_effect=mock_get)
        
        result = await fetcher.get_financial_metrics('AAPL')
        
        # PE should be set, others should be None
        assert result['pe'] == 15.0
        assert result['roe'] is None
        assert result['revenue_growth'] is None
    
    @pytest.mark.asyncio
    async def test_get_financial_metrics_no_data(self):
        """Test handling when no data is available."""
        fetcher = FMPFetcher(api_key="test-key")
        fetcher._get = AsyncMock(return_value=None)
        
        result = await fetcher.get_financial_metrics('INVALID')
        
        # All values should be None except _source
        assert result['pe'] is None
        assert result['pb'] is None
        assert result['roe'] is None
        assert result['_source'] == 'fmp'
    
    @pytest.mark.asyncio
    async def test_get_financial_metrics_empty_arrays(self):
        """Test handling of empty array responses."""
        fetcher = FMPFetcher(api_key="test-key")
        fetcher._get = AsyncMock(return_value=[])
        
        result = await fetcher.get_financial_metrics('AAPL')
        
        # Should handle empty arrays gracefully
        assert result['pe'] is None
        assert result['_source'] == 'fmp'
    
    @pytest.mark.asyncio
    async def test_get_financial_metrics_missing_fields(self):
        """Test handling of responses with missing fields."""
        fetcher = FMPFetcher(api_key="test-key")
        
        # Response has some fields but not others
        async def mock_get(endpoint, params):
            if endpoint == 'ratios':
                return [{'priceToEarningsRatio': 20.0}]  # Only PE
            return None
        
        fetcher._get = AsyncMock(side_effect=mock_get)
        
        result = await fetcher.get_financial_metrics('AAPL')
        
        assert result['pe'] == 20.0
        assert result['pb'] is None  # Missing from response


class TestGlobalFetcher:
    """Test the global fetcher singleton."""
    
    def test_get_fmp_fetcher_returns_instance(self):
        """Test that get_fmp_fetcher returns an FMPFetcher instance."""
        fetcher = get_fmp_fetcher()
        
        assert isinstance(fetcher, FMPFetcher)
    
    def test_get_fmp_fetcher_singleton(self):
        """Test that get_fmp_fetcher returns the same instance."""
        fetcher1 = get_fmp_fetcher()
        fetcher2 = get_fmp_fetcher()
        
        assert fetcher1 is fetcher2


class TestConvenienceFunction:
    """Test the fetch_fmp_metrics convenience function."""
    
    @pytest.mark.asyncio
    @patch('src.data.fmp_fetcher.get_fmp_fetcher')
    async def test_fetch_fmp_metrics_available(self, mock_get_fetcher):
        """Test convenience function when FMP is available."""
        from src.data.fmp_fetcher import fetch_fmp_metrics
        
        # Create proper mock - is_available() is NOT async
        mock_fetcher = MagicMock()
        mock_fetcher.is_available = MagicMock(return_value=True)  # Regular mock, not async
        mock_fetcher.get_financial_metrics = AsyncMock(return_value={'pe': 15.0, '_source': 'fmp'})
        mock_fetcher.__aenter__ = AsyncMock(return_value=mock_fetcher)
        mock_fetcher.__aexit__ = AsyncMock(return_value=None)
        
        mock_get_fetcher.return_value = mock_fetcher
        
        result = await fetch_fmp_metrics('AAPL')
        
        assert result['pe'] == 15.0
        mock_fetcher.get_financial_metrics.assert_called_once_with('AAPL')
    
    @pytest.mark.asyncio
    @patch('src.data.fmp_fetcher.get_fmp_fetcher')
    async def test_fetch_fmp_metrics_unavailable(self, mock_get_fetcher):
        """Test convenience function when FMP is not available."""
        from src.data.fmp_fetcher import fetch_fmp_metrics
        
        mock_fetcher = MagicMock()
        mock_fetcher.is_available.return_value = False
        mock_get_fetcher.return_value = mock_fetcher
        
        result = await fetch_fmp_metrics('AAPL')
        
        assert result is None


class TestErrorScenarios:
    """Test various error scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        """Test handling of invalid JSON response.

        Updated: After malformed JSON handling was added, the error no longer
        propagates - it returns None and logs at DEBUG level.
        """
        fetcher = FMPFetcher(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

        # Create proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)

        fetcher._session = mock_session

        # Updated: With malformed JSON handling, it should return None gracefully
        result = await fetcher._get('ratios', {'symbol': 'AAPL'})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_unexpected_response_structure(self):
        """Test handling of unexpected response structure."""
        fetcher = FMPFetcher(api_key="test-key")
        
        # Response is not a list as expected
        fetcher._get = AsyncMock(return_value={'error': 'something'})
        
        result = await fetcher.get_financial_metrics('AAPL')
        
        # Should handle gracefully and return all None
        assert result['pe'] is None
        assert result['_source'] == 'fmp'