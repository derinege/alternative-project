#!/usr/bin/env python3
"""
Dialogue Management Module for Study Buddy
Handles adaptive conversation generation based on engagement and user profile.

Author: Derin Ege Evren
Course: ELEC 491 Senior Design Project
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DialogueState(Enum):
    """Current dialogue state"""
    GREETING = "greeting"
    LEARNING = "learning"
    ASSESSMENT = "assessment"
    ENCOURAGEMENT = "encouragement"
    CLARIFICATION = "clarification"
    WRAP_UP = "wrap_up"


class EmotionalTone(Enum):
    """Emotional tone for responses"""
    ENCOURAGING = "encouraging"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
    SUPPORTIVE = "supportive"
    NEUTRAL = "neutral"
    URGENT = "urgent"


@dataclass
class DialogueContext:
    """Context for dialogue generation"""
    current_topic: str
    user_level: str
    engagement_score: float
    recent_emotions: List[str]
    session_duration: float
    interaction_count: int
    learning_goals: List[str]


class DialogueManager:
    """
    Manages adaptive dialogue generation for Study Buddy
    """
    
    def __init__(self, ollama_endpoint: str = "http://localhost:11434/api/generate"):
        """Initialize dialogue manager"""
        self.ollama_endpoint = ollama_endpoint
        self.dialogue_model = "llama3.2:1b"  # Lightweight model for real-time
        
        # Dialogue templates
        self.response_templates = self._load_response_templates()
        
        # Conversation memory
        self.conversation_history = []
        self.current_state = DialogueState.GREETING
        
        # Engagement thresholds for state transitions
        self.state_thresholds = {
            DialogueState.GREETING: 0.0,
            DialogueState.LEARNING: 0.3,
            DialogueState.ASSESSMENT: 0.6,
            DialogueState.ENCOURAGEMENT: 0.2,
            DialogueState.CLARIFICATION: 0.4,
            DialogueState.WRAP_UP: 0.8
        }
        
        print("💬 Dialogue manager initialized")
    
    def _load_response_templates(self) -> Dict:
        """Load response templates for different situations"""
        return {
            'greeting': {
                'high_engagement': [
                    "Hello! I'm excited to be your Study Buddy today! What would you like to learn about?",
                    "Hi there! I can see you're ready to dive in. What topic interests you most?",
                    "Welcome! You seem enthusiastic - let's make this study session productive!"
                ],
                'medium_engagement': [
                    "Hello! I'm your Study Buddy. How can I help you learn today?",
                    "Hi! What would you like to study together?",
                    "Welcome to our study session! What's on your mind?"
                ],
                'low_engagement': [
                    "Hello there. I'm here to help you study. What would you like to start with?",
                    "Hi! I'm your Study Buddy. Don't worry, we'll take this at your pace.",
                    "Welcome! Let's start with something comfortable. What interests you?"
                ]
            },
            'encouragement': {
                'struggling': [
                    "I can see this is challenging. That's okay - learning takes time. Let's break it down together.",
                    "Don't worry if it's not clicking yet. We can try a different approach.",
                    "It's normal to find this difficult. Let me help you understand it step by step."
                ],
                'progress': [
                    "Great job! You're making real progress. Keep going!",
                    "Excellent! I can see you're understanding this better.",
                    "Wonderful! You're really getting the hang of this."
                ],
                'excellent': [
                    "Outstanding! You're mastering this material beautifully!",
                    "Fantastic work! You're really excelling at this topic.",
                    "Amazing! You're demonstrating excellent understanding."
                ]
            },
            'clarification': {
                'confusion': [
                    "I want to make sure I understand. Could you tell me more about what's confusing?",
                    "Let me help clarify that. What specific part would you like me to explain?",
                    "I can see you might need some clarification. What would help you understand better?"
                ],
                'misunderstanding': [
                    "I think there might be a small misunderstanding. Let me explain that differently.",
                    "Let me clarify that point for you. Here's another way to think about it.",
                    "I want to make sure we're on the same page. Let me rephrase that."
                ]
            },
            'assessment': {
                'check_understanding': [
                    "Let me check if I'm explaining this clearly. What do you think so far?",
                    "How are you feeling about this topic? Any questions?",
                    "Before we move on, let's make sure you're comfortable with this."
                ],
                'quiz_style': [
                    "Here's a quick question to see how you're doing: {question}",
                    "Let me test your understanding: {question}",
                    "Quick check: {question}"
                ]
            }
        }
    
    def generate_response(self, user_input: str, engagement_level: float, 
                         user_profile, context: Dict) -> Dict:
        """Generate adaptive response based on context"""
        try:
            # Update dialogue state based on engagement
            self._update_dialogue_state(engagement_level, context)
            
            # Determine emotional tone
            tone = self._determine_emotional_tone(engagement_level, context)
            
            # Generate response based on current state
            if self.current_state == DialogueState.GREETING:
                response = self._generate_greeting_response(engagement_level, context)
            elif self.current_state == DialogueState.LEARNING:
                response = self._generate_learning_response(user_input, engagement_level, context)
            elif self.current_state == DialogueState.ENCOURAGEMENT:
                response = self._generate_encouragement_response(user_input, engagement_level, context)
            elif self.current_state == DialogueState.CLARIFICATION:
                response = self._generate_clarification_response(user_input, context)
            elif self.current_state == DialogueState.ASSESSMENT:
                response = self._generate_assessment_response(user_input, engagement_level, context)
            else:
                response = self._generate_general_response(user_input, engagement_level, context)
            
            # Enhance with LLM if needed (skip if Ollama not available)
            if context.get('use_llm', True):
                try:
                    response = self._enhance_with_llm(response, user_input, engagement_level, context)
                except Exception as e:
                    print(f"⚠️ LLM enhancement failed: {e}")
                    # Continue with template-based response
            
            # Add engagement feedback
            feedback = self._generate_engagement_feedback(engagement_level)
            
            # Determine suggested actions
            actions = self._suggest_actions(engagement_level, self.current_state)
            
            return {
                'text': response,
                'tone': tone.value,
                'state': self.current_state.value,
                'feedback': feedback,
                'actions': actions,
                'engagement_level': engagement_level
            }
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return self._fallback_response(user_input, engagement_level)
    
    def _update_dialogue_state(self, engagement_level: float, context: Dict):
        """Update dialogue state based on engagement and context"""
        # Simple state machine based on engagement and context
        if engagement_level < 0.3:
            self.current_state = DialogueState.ENCOURAGEMENT
        elif engagement_level > 0.7 and context.get('session_duration', 0) > 300:  # 5 minutes
            self.current_state = DialogueState.ASSESSMENT
        elif context.get('needs_clarification', False):
            self.current_state = DialogueState.CLARIFICATION
        elif engagement_level > 0.5:
            self.current_state = DialogueState.LEARNING
        else:
            self.current_state = DialogueState.LEARNING  # Default
    
    def _determine_emotional_tone(self, engagement_level: float, context: Dict) -> EmotionalTone:
        """Determine appropriate emotional tone"""
        if engagement_level < 0.3:
            return EmotionalTone.ENCOURAGING
        elif engagement_level > 0.8:
            return EmotionalTone.ENTHUSIASTIC
        elif context.get('user_struggling', False):
            return EmotionalTone.SUPPORTIVE
        elif engagement_level > 0.6:
            return EmotionalTone.ENCOURAGING
        else:
            return EmotionalTone.NEUTRAL
    
    def _generate_greeting_response(self, engagement_level: float, context: Dict) -> str:
        """Generate greeting response"""
        templates = self.response_templates['greeting']
        
        if engagement_level > 0.6:
            template_group = 'high_engagement'
        elif engagement_level > 0.3:
            template_group = 'medium_engagement'
        else:
            template_group = 'low_engagement'
        
        import random
        return random.choice(templates[template_group])
    
    def _generate_learning_response(self, user_input: str, engagement_level: float, context: Dict) -> str:
        """Generate learning-focused response"""
        # For now, use template-based responses
        # In a full implementation, this would integrate with educational content
        
        if engagement_level > 0.7:
            return f"That's a great question about {context.get('current_topic', 'this topic')}! Let me explain that in detail."
        elif engagement_level > 0.4:
            return f"Good question! Let me help you understand {context.get('current_topic', 'this concept')}."
        else:
            return f"I can help with that. Let's take it step by step with {context.get('current_topic', 'this topic')}."
    
    def _generate_encouragement_response(self, user_input: str, engagement_level: float, context: Dict) -> str:
        """Generate encouragement response"""
        templates = self.response_templates['encouragement']
        
        if context.get('user_struggling', False):
            return self._select_template(templates['struggling'])
        elif engagement_level > 0.7:
            return self._select_template(templates['excellent'])
        else:
            return self._select_template(templates['progress'])
    
    def _generate_clarification_response(self, user_input: str, context: Dict) -> str:
        """Generate clarification response"""
        templates = self.response_templates['clarification']
        
        if context.get('misunderstanding', False):
            return self._select_template(templates['misunderstanding'])
        else:
            return self._select_template(templates['confusion'])
    
    def _generate_assessment_response(self, user_input: str, engagement_level: float, context: Dict) -> str:
        """Generate assessment response"""
        templates = self.response_templates['assessment']
        
        # Simple assessment question (placeholder)
        if context.get('assessment_type') == 'quiz':
            question = "What do you think is the main concept here?"
            return self._select_template(templates['quiz_style']).format(question=question)
        else:
            return self._select_template(templates['check_understanding'])
    
    def _generate_general_response(self, user_input: str, engagement_level: float, context: Dict) -> str:
        """Generate general response"""
        return f"I understand. Let me help you with that. How are you feeling about our progress so far?"
    
    def _enhance_with_llm(self, base_response: str, user_input: str, 
                         engagement_level: float, context: Dict) -> str:
        """Enhance response using Ollama LLM"""
        try:
            # Create context-aware prompt
            prompt = self._create_llm_prompt(base_response, user_input, engagement_level, context)
            
            # Call Ollama
            response = requests.post(self.ollama_endpoint, json={
                "model": self.dialogue_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 100
                }
            }, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                enhanced_response = result.get('response', '').strip()
                
                # Fallback to base response if LLM fails
                if enhanced_response and len(enhanced_response) > 10:
                    return enhanced_response
                else:
                    return base_response
            else:
                return base_response
                
        except Exception as e:
            print(f"⚠️ LLM enhancement failed: {e}")
            return base_response
    
    def _create_llm_prompt(self, base_response: str, user_input: str, 
                          engagement_level: float, context: Dict) -> str:
        """Create prompt for LLM enhancement"""
        tone = self._determine_emotional_tone(engagement_level, context)
        topic = context.get('current_topic', 'general learning')
        
        prompt = f"""You are Study Buddy, an encouraging AI tutor. 

Current situation:
- User said: "{user_input}"
- Engagement level: {engagement_level:.2f} (0=low, 1=high)
- Emotional tone: {tone.value}
- Topic: {topic}

Your response should be:
- Encouraging and supportive
- Age-appropriate for students
- Helpful for learning
- Match the emotional tone: {tone.value}
- Keep it concise (1-2 sentences)

Base response: "{base_response}"

Enhanced response:"""
        
        return prompt
    
    def _generate_engagement_feedback(self, engagement_level: float) -> str:
        """Generate feedback about engagement level"""
        if engagement_level > 0.8:
            return "You're very engaged! Keep up the great work!"
        elif engagement_level > 0.6:
            return "Good engagement! You're focused and learning well."
        elif engagement_level > 0.4:
            return "You're doing well. Let me know if you need any adjustments."
        else:
            return "I notice you might be losing focus. Let's try something different!"
    
    def _suggest_actions(self, engagement_level: float, current_state: DialogueState) -> List[str]:
        """Suggest actions based on engagement and state"""
        actions = []
        
        if engagement_level < 0.3:
            actions.extend(['take_break', 'change_topic', 'use_visuals', 'simplify_content'])
        elif engagement_level > 0.8:
            actions.extend(['continue_current', 'increase_difficulty', 'explore_deeper'])
        else:
            actions.extend(['monitor_engagement', 'maintain_pace'])
        
        # State-specific actions
        if current_state == DialogueState.ASSESSMENT:
            actions.extend(['quiz_questions', 'check_understanding'])
        elif current_state == DialogueState.CLARIFICATION:
            actions.extend(['provide_examples', 'rephrase_explanation'])
        
        return actions[:3]  # Limit to 3 actions
    
    def _select_template(self, templates: List[str]) -> str:
        """Select random template from list"""
        import random
        return random.choice(templates)
    
    def _fallback_response(self, user_input: str, engagement_level: float) -> Dict:
        """Fallback response when generation fails"""
        if engagement_level < 0.3:
            response = "I'm here to help you learn. Let's try a different approach."
            tone = EmotionalTone.ENCOURAGING
        else:
            response = "I understand. Let me help you with that."
            tone = EmotionalTone.NEUTRAL
        
        return {
            'text': response,
            'tone': tone.value,
            'state': 'fallback',
            'feedback': 'Using fallback response',
            'actions': ['monitor_engagement'],
            'engagement_level': engagement_level
        }
    
    def add_to_conversation_history(self, user_input: str, bot_response: str, engagement_level: float):
        """Add interaction to conversation history"""
        interaction = {
            'timestamp': time.time(),
            'user_input': user_input,
            'bot_response': bot_response,
            'engagement_level': engagement_level,
            'state': self.current_state.value
        }
        
        self.conversation_history.append(interaction)
        
        # Keep only recent history (last 20 interactions)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return {'error': 'No conversation history'}
        
        recent_interactions = self.conversation_history[-10:]
        
        return {
            'total_interactions': len(self.conversation_history),
            'recent_interactions': len(recent_interactions),
            'average_engagement': sum(i['engagement_level'] for i in recent_interactions) / len(recent_interactions),
            'current_state': self.current_state.value,
            'recent_states': [i['state'] for i in recent_interactions[-5:]]
        }


def main():
    """Test dialogue manager"""
    print("💬 Dialogue Manager Test")
    print("=" * 40)
    
    manager = DialogueManager()
    
    # Test response generation
    user_input = "I'm having trouble understanding this concept"
    engagement_level = 0.3
    context = {
        'current_topic': 'mathematics',
        'session_duration': 600,  # 10 minutes
        'user_struggling': True,
        'use_llm': False  # Test without LLM first
    }
    
    response = manager.generate_response(user_input, engagement_level, None, context)
    
    print(f"User input: {user_input}")
    print(f"Engagement level: {engagement_level}")
    print(f"Bot response: {response['text']}")
    print(f"Tone: {response['tone']}")
    print(f"State: {response['state']}")
    print(f"Feedback: {response['feedback']}")
    print(f"Suggested actions: {response['actions']}")
    
    # Test conversation history
    manager.add_to_conversation_history(user_input, response['text'], engagement_level)
    summary = manager.get_conversation_summary()
    print(f"\nConversation summary: {summary}")


if __name__ == "__main__":
    main()
