"""
memory/memory_manager.py
------------------------
Memory Manager — stores and retrieves per-character reference images and descriptions.

Responsibilities
----------------
- Maintain a simple in-memory dictionary keyed by character name
- Store the latest reference image for each character
- Store character descriptions (write-once: first description wins)
- Retrieve reference images and descriptions on demand

Public interface
----------------
    MemoryManager.get_reference_images(characters)      -> dict[str, PIL.Image]
    MemoryManager.update_memory(characters, image, descriptions)
    MemoryManager.get_character_descriptions(characters) -> dict[str, str]

No embeddings. No similarity logic. No external dependencies.
"""

import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Minimal per-character memory store.

    Internal storage
    ----------------
    _store : dict[str, dict]
        {
            character_name: {
                "description":     str,
                "reference_image": PIL.Image,
            },
            ...
        }

    Behaviour
    ---------
    - Characters are created on first update.
    - reference_image is always overwritten with the latest image.
    - description is written once; subsequent updates are ignored.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        logger.info("MemoryManager initialised (empty)")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_reference_images(
        self, characters: list[str]
    ) -> dict[str, Image.Image]:
        """
        Return stored reference images for the requested characters.

        Only characters that have an existing entry with a reference image
        are included in the result. Missing characters are silently skipped.

        Parameters
        ----------
        characters : List of character names to look up.

        Returns
        -------
        Mapping of character_name -> PIL.Image for those that exist.
        """
        result: dict[str, Image.Image] = {}

        if not characters:
            logger.debug(
                "get_reference_images | no characters provided — returning empty dict"
            )
            return result

        for name in characters:
            entry = self._store.get(name)
            if entry is not None and entry.get("reference_image") is not None:
                result[name] = entry["reference_image"].copy()
            else:
                logger.debug(
                    "get_reference_images | no entry for %r — skipping", name
                )

        logger.info(
            "get_reference_images | requested=%d  found=%d",
            len(characters), len(result),
        )
        return result

    def update_memory(
        self,
        characters: list[str],
        image: Image.Image,
        descriptions: dict[str, str],
    ) -> None:
        """
        Update memory for each character in the list.

        For every character:
        - reference_image is always overwritten with the provided image.
        - description is written only if no description is stored yet
          (first-write-wins policy).

        Parameters
        ----------
        characters   : List of character names present in this scene.
        image        : The selected scene image, stored as reference for all
                       characters in this scene (architecture limitation noted
                       in ARCHITECTURE.md).
        descriptions : Mapping of character_name -> description string.
                       Only used when creating a new entry.
        """
        if not characters: 
            logger.debug("update_memory | empty character list — nothing to do") 
            return 

        if image is None: 
            raise ValueError("update_memory | image cannot be None")

        descriptions = descriptions or {}

        for name in characters:
            if name not in self._store:
                # New character — initialise entry
                description = descriptions.get(name, "")
                
                self._store[name] = {
                    "description":     description,
                    "reference_image": image.copy(),
                }
                logger.info(
                    "update_memory | new character %r  description=%r",
                    name, description[:60] if description else "",
                )
            else:
                # Existing character — overwrite image, keep description
                self._store[name]["reference_image"] = image.copy()

                # Write description only if not already present
                if not self._store[name].get("description"):
                    description = descriptions.get(name)
                    if description:
                        self._store[name]["description"] = description
                    logger.debug(
                        "update_memory | set description for %r (was empty)", name
                    )

                logger.info(
                    "update_memory | updated reference_image for %r", name
                )

    def get_character_descriptions(
        self, characters: list[str]
    ) -> dict[str, str]:
        """
        Collect non-empty stored descriptions for the given names.

        Parameters
        ----------
        characters : list[str]
            Names to query in iteration order.

        Returns
        -------
        dict[str, str]
            Entries where ``entry.get("description")`` is truthy.

        Notes
        -----
        Skips missing names or empty descriptions with debug logs; ends with
        info count line.

        Edge cases
        ----------
        Duplicate names in input would duplicate work but preserve last write in
        the result dict.
        """
        result: dict[str, str] = {}

        for name in characters:
            entry = self._store.get(name)
            if entry is not None and entry.get("description"):
                result[name] = entry["description"]
            else:
                logger.debug(
                    "get_character_descriptions | no description for %r — skipping",
                    name,
                )

        logger.info(
            "get_character_descriptions | requested=%d  found=%d",
            len(characters), len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Diagnostics (not part of public contract — useful for logging)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the count of character keys in the internal store.

        Parameters
        ----------
        None

        Returns
        -------
        int
            ``len(self._store)``.

        Notes
        -----
        O(1) delegation to builtin ``len`` on the mapping.

        Edge cases
        ----------
        Empty manager returns ``0``.
        """
        return len(self._store)

    def __repr__(self) -> str:
        """
        Render a concise representation listing tracked character names.

        Parameters
        ----------
        None

        Returns
        -------
        str
            ``MemoryManager(characters=[...])`` with current keys.

        Notes
        -----
        Copies keys to a list for display order.

        Edge cases
        ----------
        Large numbers of names produce a long single-line string.
        """
        names = list(self._store.keys())
        return f"MemoryManager(characters={names})"

    def has_character(self, name: str) -> bool:
        """
        Report whether ``name`` exists in the memory store.

        Parameters
        ----------
        name : str
            Character name key.

        Returns
        -------
        bool
            ``True`` if ``name in self._store``.

        Notes
        -----
        Membership test only; does not validate image or description presence.

        Edge cases
        ----------
        Empty string is a valid key if inserted by callers.
        """
        return name in self._store