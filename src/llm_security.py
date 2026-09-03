"""Realiza a inspeção de seguranca 

Este script realiza a inspeção de segurança  
"""

from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import PromptInjection, TokenLimit, InvisibleText, Secrets
from llm_guard.input_scanners.prompt_injection import MatchType
from llm_guard.output_scanners import Sensitive, Relevance


class LlmSecurity:
    """
    Classe responsável por gerenciar a segurança de entrada e saída das LLMs.
    """
    def __init__(self):
        self.input_scanners = [
            InvisibleText(),
            Secrets(),
            TokenLimit(limit=4096),
            PromptInjection(
                threshold=0.5,
                match_type=MatchType.FULL
            ),
        ]

        self.output_scanners = [
            Sensitive(),
            Relevance(),
        ]

    def validate_input(self, text: str) -> str:
        """
        Valida e limpa o prompt do usuário antes de enviá-lo ao modelo.
        
        Args:
            text (str): O prompt original do usuário.
            
        Returns:
            str: O prompt sanitizado.
            
        Raises:
            ValueError: Se qualquer scanner de segurança for violado.
        """
        sanitized_prompt, results_valid, results_score = scan_prompt(
            self.input_scanners,
            text
        )

        if any(not result for result in results_valid.values()):
            raise ValueError(
                f"Entrada bloqueada por segurança. Scores: {results_score}"
            )

        return sanitized_prompt

    def validate_output(self, prompt: str, answer: str) -> str:
        """
        Valida e limpa a resposta gerada pelo modelo antes de exibi-la.
        
        Args:
            prompt (str): O prompt original enviado.
            answer (str): A resposta gerada pela LLM.
            
        Returns:
            str: A resposta sanitizada.
            
        Raises:
            ValueError: Se a saída violar os critérios de segurança ou relevância.
        """

        sanitized_output, results_valid, results_score = scan_output(
            self.output_scanners,
            prompt,
            answer
        )

        if any(not result for result in results_valid.values()):
            raise ValueError(
                f"Saída bloqueada por segurança. Scores: {results_score}"
            )

        return sanitized_output