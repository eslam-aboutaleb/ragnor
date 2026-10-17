"""Test execution of the starter framework recipes."""

from __future__ import annotations


from examples.recipes import customer_service_faq, multi_collection_diversity, policy_knowledge


async def test_policy_knowledge_recipe() -> None:
    """Run Recipe 1: Policy knowledge with page citations and revision updates."""
    await policy_knowledge.run_recipe()
    profile = policy_knowledge.production_profile()
    assert profile["profile"] == "production-policy-compliance"
    assert profile["vector_store_provider"] == "pgvector"


async def test_customer_service_faq_recipe() -> None:
    """Run Recipe 2: Customer service knowledge with product/locale filters and deletion."""
    await customer_service_faq.run_recipe()
    profile = customer_service_faq.production_profile()
    assert profile["profile"] == "production-customer-service-faqs"
    assert profile["vector_store_provider"] == "qdrant"


async def test_multi_collection_diversity_recipe() -> None:
    """Run Recipe 3: Multi-collection search with source diversity."""
    await multi_collection_diversity.run_recipe()
    profile = multi_collection_diversity.production_profile()
    assert profile["profile"] == "production-multi-collection-federation"
    assert len(profile["collections"]) == 2
