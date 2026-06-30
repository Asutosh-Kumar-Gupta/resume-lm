'use server';
import {
  APICallError,
  generateObject,
  RetryError,
  type LanguageModelUsage,
  type LanguageModelV1,
  type TelemetrySettings,
} from 'ai';
import { z } from 'zod';
import { RESUME_FORMATTER_SYSTEM_MESSAGE } from "@/lib/prompts";
import { type AIConfig } from '@/utils/ai-tools';
import { getSubscriptionPlan } from '@/utils/actions/stripe/actions';
import { sanitizeUnknownStrings } from '@/lib/utils';
import { withTaskModel } from '@/lib/ai/task-models';
import {
  finishAIUsageRequest,
  startAIUsageRequest,
} from '@/lib/ai/usage-ledger';

async function runTrackedAIRequest<T extends { usage?: LanguageModelUsage }>(
  input: {
    route: string;
    userId: string;
    isPro: boolean;
    config?: AIConfig;
  },
  task: (model: LanguageModelV1, telemetry: TelemetrySettings) => Promise<T>
) {
  const { model, usageEventId, telemetry } = await startAIUsageRequest(input);

  try {
    const result = await task(model, telemetry);
    await finishAIUsageRequest({
      usageEventId,
      status: 'succeeded',
      usage: result.usage,
    });
    return result;
  } catch (error) {
    await finishAIUsageRequest({
      usageEventId,
      status: 'failed',
      errorCode: error instanceof Error ? error.message : 'ai_request_failed',
    });
    throw error;
  }
}

function isTemporaryCapacityError(error: unknown): boolean {
  const apiError = APICallError.isInstance(error)
    ? error
    : RetryError.isInstance(error) && APICallError.isInstance(error.lastError)
      ? error.lastError
      : undefined;

  return apiError?.statusCode === 503;
}

// TEXT RESUME -> PROFILE
export async function formatProfileWithAI(
  userMessages: string,
  config?: AIConfig
) {
  const { plan, id } = await getSubscriptionPlan(true);
  const isPro = plan === 'pro';
  const primaryConfig = withTaskModel({
    task: "structuredExtraction",
    isPro,
    config,
  });

  const runExtraction = (candidateConfig: AIConfig) =>
    runTrackedAIRequest({
      route: 'actions.profiles.formatProfileWithAI',
      userId: id,
      isPro,
      config: candidateConfig,
    }, (aiClient, telemetry) => generateObject({
        model: aiClient as LanguageModelV1,
        experimental_telemetry: telemetry,
        maxRetries: 1,
        schema: z.object({
          content: z.object({
            first_name: z.string().optional(),
            last_name: z.string().optional(),
            email: z.string().optional(),
            phone_number: z.string().optional(),
            location: z.string().optional(),
            website: z.string().optional(),
            linkedin_url: z.string().optional(),
            github_url: z.string().optional(),
            work_experience: z.array(z.object({
              company: z.string(),
              position: z.string(),
              date: z.string(),
              location: z.string().optional(),
              description: z.array(z.string()),
              technologies: z.array(z.string()).optional()
            })).optional(),
            education: z.array(z.object({
              school: z.string(),
              degree: z.string(),
              field: z.string(),
              date: z.string(),
              location: z.string().optional(),
              gpa: z.string().optional(),
              achievements: z.array(z.string()).optional()
            })).optional(),
            skills: z.array(z.object({
              category: z.string(),
              items: z.array(z.string())
            })).optional(),
            projects: z.array(z.object({
              name: z.string(),
              description: z.array(z.string()),
              technologies: z.array(z.string()).optional(),
              date: z.string().optional(),
              url: z.string().optional(),
              github_url: z.string().optional()
            })).optional()
          })
        }),
        prompt: `Please analyze this resume text and extract all relevant information into a structured profile format. 
                Include all sections (personal info, work experience, education, skills, projects) if present.
                Ensure all arrays (like description, technologies, achievements) are properly formatted as arrays.
                For any missing or unclear information, use optional fields rather than making assumptions.
  
                Resume Text:
  ${userMessages}`,
        // Use custom prompt if provided in config, otherwise fall back to default
        system: config?.customPrompts?.resumeFormatter 
          ?? (RESUME_FORMATTER_SYSTEM_MESSAGE.content as string),
      }));

  let result;
  try {
    result = await runExtraction(primaryConfig);
  } catch (error) {
    if (!isTemporaryCapacityError(error)) {
      throw error;
    }

    result = await runExtraction({
      ...primaryConfig,
      model: 'gemini-2.5-flash-lite',
    });
  }

  return sanitizeUnknownStrings(result.object.content);
}
