"""
Unit tests for the URL matrix fixture module.

Validates that:
1. Every UrlEntry carries all four required metadata fields.
2. All field values are within the declared valid sets.
3. There are no duplicate URLs.
4. The matrix represents a meaningful cross-section of languages,
   rendering types, and content categories.
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.fixtures.url_matrix import (
    REQUIRED_FIELDS,
    URL_MATRIX,
    VALID_CATEGORIES,
    VALID_LANGUAGES,
    VALID_RENDERING_TYPES,
    UrlEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY_IDS: list[str] = [
    f"{i}_{e.language}_{e.rendering_type}_{e.category}"
    for i, e in enumerate(URL_MATRIX)
]


# ---------------------------------------------------------------------------
# 1. Required fields — every entry must expose all four fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Every UrlEntry must carry all four required metadata fields."""

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_entry_is_url_entry_instance(self, entry: UrlEntry) -> None:
        assert isinstance(entry, UrlEntry)

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_entry_has_all_required_fields(self, entry: UrlEntry) -> None:
        """Programmatic check via dataclasses.asdict mirrors REQUIRED_FIELDS."""
        entry_dict = dataclasses.asdict(entry)
        missing = REQUIRED_FIELDS - set(entry_dict.keys())
        assert not missing, (
            f"Entry {entry.url!r} is missing required fields: {missing}"
        )

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_url_field_is_non_empty_string(self, entry: UrlEntry) -> None:
        assert isinstance(entry.url, str)
        assert entry.url.strip()

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_category_field_is_non_empty_string(self, entry: UrlEntry) -> None:
        assert isinstance(entry.category, str)
        assert entry.category.strip()

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_language_field_is_non_empty_string(self, entry: UrlEntry) -> None:
        assert isinstance(entry.language, str)
        assert entry.language.strip()

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_rendering_type_field_is_non_empty_string(self, entry: UrlEntry) -> None:
        assert isinstance(entry.rendering_type, str)
        assert entry.rendering_type.strip()


# ---------------------------------------------------------------------------
# 2. Field value validity — all values within declared enumerations
# ---------------------------------------------------------------------------


class TestFieldValues:
    """All field values must belong to the declared valid sets."""

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_url_starts_with_https(self, entry: UrlEntry) -> None:
        assert entry.url.startswith("https://"), (
            f"URL {entry.url!r} must start with 'https://'"
        )

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_language_is_valid(self, entry: UrlEntry) -> None:
        assert entry.language in VALID_LANGUAGES, (
            f"language {entry.language!r} not in {sorted(VALID_LANGUAGES)}"
        )

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_rendering_type_is_valid(self, entry: UrlEntry) -> None:
        assert entry.rendering_type in VALID_RENDERING_TYPES, (
            f"rendering_type {entry.rendering_type!r} not in "
            f"{sorted(VALID_RENDERING_TYPES)}"
        )

    @pytest.mark.parametrize("entry", URL_MATRIX, ids=_ENTRY_IDS)
    def test_category_is_valid(self, entry: UrlEntry) -> None:
        assert entry.category in VALID_CATEGORIES, (
            f"category {entry.category!r} not in {sorted(VALID_CATEGORIES)}"
        )


# ---------------------------------------------------------------------------
# 3. No duplicate URLs
# ---------------------------------------------------------------------------


class TestNoDuplicates:
    """The matrix must not contain the same URL twice."""

    def test_no_duplicate_urls(self) -> None:
        urls = [e.url for e in URL_MATRIX]
        duplicates = {u for u in urls if urls.count(u) > 1}
        assert not duplicates, f"Duplicate URLs found in URL_MATRIX: {duplicates}"

    def test_url_count_matches_set_size(self) -> None:
        urls = [e.url for e in URL_MATRIX]
        assert len(urls) == len(set(urls)), (
            "URL_MATRIX contains duplicate entries"
        )


# ---------------------------------------------------------------------------
# 4. Matrix coverage — meaningful cross-section of the test space
# ---------------------------------------------------------------------------


class TestMatrixCoverage:
    """The matrix must represent a useful cross-section of scraping scenarios."""

    def test_matrix_has_at_least_ten_entries(self) -> None:
        assert len(URL_MATRIX) >= 10, (
            f"URL_MATRIX has only {len(URL_MATRIX)} entries; expected >= 10"
        )

    def test_matrix_includes_english_content(self) -> None:
        en_entries = [e for e in URL_MATRIX if e.language == "en"]
        assert len(en_entries) >= 1, "URL_MATRIX must include at least one English URL"

    def test_matrix_includes_korean_content(self) -> None:
        ko_entries = [e for e in URL_MATRIX if e.language == "ko"]
        assert len(ko_entries) >= 1, "URL_MATRIX must include at least one Korean URL"

    def test_matrix_includes_static_rendering_type(self) -> None:
        static_entries = [e for e in URL_MATRIX if e.rendering_type == "static"]
        assert len(static_entries) >= 1, (
            "URL_MATRIX must include at least one 'static' rendering_type entry"
        )

    def test_matrix_includes_headers_rotation_rendering_type(self) -> None:
        hr_entries = [e for e in URL_MATRIX if e.rendering_type == "headers_rotation"]
        assert len(hr_entries) >= 1, (
            "URL_MATRIX must include at least one 'headers_rotation' rendering_type entry"
        )

    def test_matrix_includes_browser_rendering_type(self) -> None:
        browser_entries = [e for e in URL_MATRIX if e.rendering_type == "browser"]
        assert len(browser_entries) >= 1, (
            "URL_MATRIX must include at least one 'browser' rendering_type entry"
        )

    def test_matrix_covers_all_rendering_types(self) -> None:
        observed = {e.rendering_type for e in URL_MATRIX}
        assert observed == VALID_RENDERING_TYPES, (
            f"URL_MATRIX does not cover all rendering types. "
            f"Missing: {VALID_RENDERING_TYPES - observed}"
        )

    def test_matrix_covers_both_languages(self) -> None:
        observed = {e.language for e in URL_MATRIX}
        assert observed == VALID_LANGUAGES, (
            f"URL_MATRIX does not cover all languages. "
            f"Missing: {VALID_LANGUAGES - observed}"
        )

    def test_matrix_includes_multiple_categories(self) -> None:
        categories = {e.category for e in URL_MATRIX}
        assert len(categories) >= 2, (
            f"URL_MATRIX uses only {len(categories)} category/categories; "
            "expected >= 2 for meaningful coverage"
        )


# ---------------------------------------------------------------------------
# 5. UrlEntry immutability (frozen dataclass)
# ---------------------------------------------------------------------------


class TestUrlEntryImmutability:
    """UrlEntry must be immutable (frozen dataclass)."""

    def test_cannot_mutate_url(self) -> None:
        entry = URL_MATRIX[0]
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            entry.url = "https://mutated.example.com/"  # type: ignore[misc]

    def test_cannot_mutate_language(self) -> None:
        entry = URL_MATRIX[0]
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            entry.language = "ja"  # type: ignore[misc]

    def test_url_entry_is_hashable(self) -> None:
        entry = URL_MATRIX[0]
        h = hash(entry)
        assert isinstance(h, int)

    def test_url_entries_can_be_added_to_set(self) -> None:
        entries_set = set(URL_MATRIX)
        assert len(entries_set) == len(URL_MATRIX)


# ---------------------------------------------------------------------------
# 6. REQUIRED_FIELDS constant
# ---------------------------------------------------------------------------


class TestRequiredFieldsConstant:
    """Verify REQUIRED_FIELDS has the expected value."""

    def test_required_fields_is_frozenset(self) -> None:
        assert isinstance(REQUIRED_FIELDS, frozenset)

    def test_required_fields_contains_url(self) -> None:
        assert "url" in REQUIRED_FIELDS

    def test_required_fields_contains_category(self) -> None:
        assert "category" in REQUIRED_FIELDS

    def test_required_fields_contains_language(self) -> None:
        assert "language" in REQUIRED_FIELDS

    def test_required_fields_contains_rendering_type(self) -> None:
        assert "rendering_type" in REQUIRED_FIELDS

    def test_required_fields_has_exactly_four_members(self) -> None:
        assert len(REQUIRED_FIELDS) == 4


# ---------------------------------------------------------------------------
# 7. UrlEntry.as_dict() helper
# ---------------------------------------------------------------------------


class TestUrlEntryAsDict:
    """UrlEntry.as_dict() must return a plain dict with all required keys."""

    def test_as_dict_returns_dict(self) -> None:
        entry = URL_MATRIX[0]
        assert isinstance(entry.as_dict(), dict)

    def test_as_dict_contains_all_required_keys(self) -> None:
        entry = URL_MATRIX[0]
        result = entry.as_dict()
        assert set(result.keys()) >= REQUIRED_FIELDS

    def test_as_dict_values_match_entry_fields(self) -> None:
        entry = URL_MATRIX[0]
        result = entry.as_dict()
        assert result["url"] == entry.url
        assert result["category"] == entry.category
        assert result["language"] == entry.language
        assert result["rendering_type"] == entry.rendering_type
